from models import HistoryEvent


def format_event_text(event: HistoryEvent) -> str:
    """Convert HistoryEvent to readable chat-style text."""
    snapshot = event.snapshot_data or {}

    if event.event_type == "session_created":
        provider = snapshot.get("model_provider", "unknown")
        return f"Session started with {provider} model."
    
    elif event.event_type == "product_image_upload":
        return "User uploaded product image."
    
    elif event.event_type == "product_image_analyzed":
        product_type = snapshot.get("product_type", "not specified")
        product_category = snapshot.get("product_category", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Product analyzed by {provider}.\n"
            f"Detected type: {product_type}, category: {product_category}."
        )
    
    elif event.event_type == "moodboard_upload":
        return "User uploaded moodboard image."
    
    elif event.event_type == "moodboard_image_analyzed":
        visual_style = snapshot.get("visual_style", "not specified")
        mood_atmo = snapshot.get("mood_atmosphere", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Moodboard image analyzed by {provider}.\n"
            f"Visual style: {visual_style}\n"
            f"Mood: {mood_atmo}"
        )
    
    elif event.event_type == "user_vision_submitted":
        preview_text = snapshot.get("preview_text", "")
        if preview_text:
            return (
                "User submitted their vision.\n"
                f"Preview: {preview_text}"
            )
        return "User submitted their vision."
    
    elif event.event_type == "vision_parsed":
        focus_subject = snapshot.get("focus_subject", "not specified")
        setting = snapshot.get("setting", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Vision parsed and structured by {provider}.\n"
            f"Focus subject: {focus_subject}\n"
            f"Setting: {setting}"
        )
    
    elif event.event_type == "prompt_built":
        focus_slider = snapshot.get("focus_slider", "unknown")
        used_rag_examples = snapshot.get("used_rag_examples", False)
        provider = snapshot.get("model_provider", "unknown")
        rag_status = "Yes" if used_rag_examples else "No"
        return (
            f"Advertising prompt built by {provider}.\n"
            f"Focus slider: {focus_slider}\n"
            f"Prompt examples used: {rag_status}"
        )
    
    elif event.event_type == "prompt_refined":
        focus_slider = snapshot.get("focus_slider", "unknown")
        refinement_count = snapshot.get("refinement_count", "unknown")
        used_rag_examples = snapshot.get("used_rag_examples", False)
        provider = snapshot.get("model_provider", "unknown")
        rag_status = "Yes" if used_rag_examples else "No"
        return (
            f"Advertising prompt refined by {provider}.\n"
            f"Focus slider: {focus_slider}\n"
            f"Refinement count: {refinement_count}\n"
            f"Prompt examples used: {rag_status}"
        )
    
    elif event.event_type == "prompt_refinement_request":
        feedback = snapshot.get("user_feedback")
        focus_slider = snapshot.get("focus_slider")
        
        parts = ["User requested prompt refinement."]
        if feedback:
            parts.append(f"\nFeedback: {feedback}")
        if focus_slider is not None:
            parts.append(f"\nFocus slider: {focus_slider}")
        
        return "".join(parts)
    
    elif event.event_type == "image_model_chosen":
        model = snapshot.get("image_model", "unknown")
        return f"Image generation model chosen: {model}."
    
    elif event.event_type == "reference_image_upload":
        return "User uploaded reference image."
    
    elif event.event_type == "image_generated":
        model = snapshot.get("model", "unknown")
        return f"Ad image generated successfully with {model} model."
    
    else:
        return f"Unknown event: {event.event_type}"
