from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .ocr import recognise_handwriting


def dashboard(request):
    context = {
        "stats": [
            {"label": "Scripts processed", "value": "128", "delta": "+12 this week"},
            {"label": "Awaiting review", "value": "18", "delta": "Teacher action needed"},
            {"label": "Examinations", "value": "06", "delta": "2 currently active"},
            {"label": "OCR accuracy", "value": "Preview", "delta": "Verify every extraction"},
        ],
        "recent_scripts": [
            {"student": "Aarav Sharma", "exam": "Computer Science · Mid-term", "status": "Ready for review", "tone": "green", "date": "Today, 10:42 AM"},
            {"student": "Ishita Das", "exam": "Mathematics · Unit Test 3", "status": "OCR processing", "tone": "blue", "date": "Today, 10:18 AM"},
            {"student": "Rohan Mehta", "exam": "Physics · Practice Paper", "status": "Needs attention", "tone": "amber", "date": "Yesterday"},
        ],
    }
    return render(request, "evaluation/dashboard.html", context)


@require_http_methods(["GET", "POST"])
def upload_answer(request):
    if request.method == "GET":
        return render(request, "evaluation/upload.html")

    uploaded = request.FILES.get("answer_sheet")
    if not uploaded:
        messages.error(request, "Choose an image before starting OCR.")
        return redirect("upload_answer")

    if uploaded.size > 10 * 1024 * 1024:
        messages.error(request, "The image must be smaller than 10 MB.")
        return redirect("upload_answer")

    if uploaded.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        messages.error(request, "Please upload a JPG, PNG, or WEBP image.")
        return redirect("upload_answer")

    try:
        extracted_text = recognise_handwriting(uploaded)
    except Exception as exc:
        return render(request, "evaluation/upload.html", {
            "error": "OCR could not process this image. Check the setup instructions and try a clear, cropped handwriting sample.",
            "technical_error": str(exc),
        })

    return render(request, "evaluation/review.html", {
        "extracted_text": extracted_text,
        "filename": uploaded.name,
    })
