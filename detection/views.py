import base64
import os

import cv2
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from .forms import ImageUploadForm
from .services import InvalidImageError, analyse_image


_model = None


def get_model():
    global _model
    if _model is None:
        from keras.models import load_model
        _model = load_model(os.path.join(settings.BASE_DIR, 'cyst_detection_model.h5'))
    return _model


def upload_and_detect(request):
    context = {}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            relative_path = default_storage.save('uploads/input-image', request.FILES['image'])
            full_path = default_storage.path(relative_path)
            try:
                result = analyse_image(full_path, get_model())
                ok, encoded = cv2.imencode('.jpg', result.annotated_image)
                if not ok:
                    raise InvalidImageError('The result image could not be encoded.')
                context.update({
                    'label': result.label,
                    'confidence': f'{result.confidence:.1%}',
                    'contour_count': result.contour_count,
                    'result_image': base64.b64encode(encoded).decode('ascii'),
                })
            except InvalidImageError as exc:
                context['error'] = str(exc)
            finally:
                default_storage.delete(relative_path)
    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, 'result.html' if 'label' in context else 'upload.html', context)
