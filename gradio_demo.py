"""
Gradio Demo for ArXiv Classifier API.

This demo provides a web interface to interact with the Django-based
ArXiv classifier API for classifying research article abstracts.

Usage:
    1. Start the Django server: cd backend && python manage.py runserver
    2. Run this demo: python gradio_demo.py
"""
import gradio as gr
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health():
    """Check if the API is available and get model info."""
    try:
        response = requests.get(f"{API_BASE_URL}/health/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def classify_abstract(abstract: str, threshold: float = 0.5) -> tuple[str, str]:
    """
    Classify a research abstract using the API.
    
    Args:
        abstract: The research article abstract text
        threshold: Minimum probability threshold for displaying predictions
        
    Returns:
        Tuple of (formatted results, raw JSON response)
    """
    if not abstract.strip():
        return "Please enter an abstract to classify.", ""
    
    # Check API health first
    is_healthy, health_data = check_api_health()
    if not is_healthy:
        error_msg = health_data.get("error", "Unknown error")
        return f"API not available: {error_msg}\n\nMake sure the Django server is running:\n```\ncd backend && python manage.py runserver\n```", ""
    
    # Prepare request
    payload = {"abstracts": [abstract]}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            error_detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            return f"API Error ({response.status_code}):\n{json.dumps(error_detail, indent=2)}", ""
        
        data = response.json()
        
        # Format results
        result = data.get("results", [{}])[0]
        predictions = result.get("predictions", [])
        model_version = data.get("model_version", "unknown")
        
        # Filter by threshold and sort by probability
        filtered_predictions = [p for p in predictions if p["probability"] >= threshold]
        sorted_predictions = sorted(filtered_predictions, key=lambda x: x["probability"], reverse=True)
        
        if not sorted_predictions:
            formatted_output = f"**No predictions above threshold ({threshold})**\n\n"
            formatted_output += f"_Model: {model_version}_"
        else:
            formatted_output = "## Predicted Categories\n\n"
            
            for i, pred in enumerate(sorted_predictions, 1):
                label = pred["label"]
                prob = pred["probability"]
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                formatted_output += f"**{i}. {label}**\n"
                formatted_output += f"   `{bar}` {prob:.1%}\n\n"
            
            formatted_output += f"---\n_Model: {model_version} | Threshold: {threshold}_"
        
        # Pretty print raw JSON
        raw_json = json.dumps(data, indent=2)
        
        return formatted_output, raw_json
        
    except requests.exceptions.Timeout:
        return "Request timed out. The model may be loading or the server is busy.", ""
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}", ""
    except json.JSONDecodeError:
        return "Failed to parse API response as JSON.", ""


def get_api_status():
    """Get formatted API status for display."""
    is_healthy, data = check_api_health()
    
    if is_healthy:
        status = data.get("status", "unknown")
        model_loaded = data.get("model_loaded", False)
        model_type = data.get("model_type", "N/A")
        version = data.get("version", "N/A")
        
        return f"""### API Status: {status.capitalize()}

| Property | Value |
|----------|-------|
| Model Loaded | {'Yes' if model_loaded else 'No'} |
| Model Type | {model_type} |
| API Version | {version} |
"""
    else:
        error = data.get("error", "Unknown error")
        return f"""### API Not Available

**Error:** {error}

Make sure the Django server is running:
```bash
cd backend && python manage.py runserver
```
"""


# Example abstracts for testing
EXAMPLE_ABSTRACTS = [
    """We propose a novel deep learning architecture for image classification 
that combines convolutional neural networks with attention mechanisms. 
Our model achieves state-of-the-art performance on ImageNet and CIFAR-10 
benchmarks while requiring significantly fewer parameters than existing 
approaches. We demonstrate the effectiveness of our attention module 
through extensive ablation studies and visualization of learned features.""",

    """This paper presents a comprehensive analysis of quantum error correction 
codes in the presence of correlated noise. We derive new bounds on the 
threshold error rate for fault-tolerant quantum computation and propose 
an improved decoding algorithm that outperforms existing methods. Our 
results have important implications for the practical realization of 
large-scale quantum computers.""",

    """We study the asymptotic behavior of solutions to nonlinear partial 
differential equations arising in fluid dynamics. Using techniques from 
functional analysis and spectral theory, we establish new regularity 
results for the Navier-Stokes equations in three dimensions. Our approach 
provides insights into the mathematical structure of turbulent flows.""",
]


# Build Gradio interface
with gr.Blocks(
    title="ArXiv Classifier Demo",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
        .gradio-container { max-width: 1200px !important; }
        .status-box { font-family: monospace; }
    """
) as demo:
    gr.Markdown("""
    # ArXiv Paper Classifier
    
    Classify research paper abstracts into ArXiv categories.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            abstract_input = gr.Textbox(
                label="Abstract",
                placeholder="Paste your research paper abstract here...",
                lines=8,
                max_lines=15,
            )
            
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Probability Threshold",
                    info="Only show predictions above this confidence level"
                )
                
                classify_btn = gr.Button(
                    "Classify",
                    variant="primary",
                    size="lg"
                )
            
            # Example buttons
            gr.Markdown("### Examples")
            with gr.Row():
                example_btns = []
                for i, label in enumerate(["Deep Learning", "Quantum Computing", "Math/Physics"]):
                    btn = gr.Button(label, size="sm")
                    example_btns.append(btn)
        
        with gr.Column(scale=2):
            # Output section
            results_output = gr.Markdown(
                label="Classification Results",
                value="*Results will appear here after classification*"
            )
    
    with gr.Accordion("Raw API Response", open=False):
        raw_json_output = gr.Code(
            label="JSON Response",
            language="json",
        )
    
    with gr.Accordion("API Status", open=False):
        status_output = gr.Markdown()
        refresh_btn = gr.Button("Refresh Status", size="sm")
    
    # Event handlers
    classify_btn.click(
        fn=classify_abstract,
        inputs=[abstract_input, threshold_slider],
        outputs=[results_output, raw_json_output]
    )
    
    abstract_input.submit(
        fn=classify_abstract,
        inputs=[abstract_input, threshold_slider],
        outputs=[results_output, raw_json_output]
    )
    
    refresh_btn.click(
        fn=get_api_status,
        outputs=[status_output]
    )
    
    # Example button handlers
    for i, btn in enumerate(example_btns):
        btn.click(
            fn=lambda idx=i: EXAMPLE_ABSTRACTS[idx],
            outputs=[abstract_input]
        )
    
    # Load status on start
    demo.load(fn=get_api_status, outputs=[status_output])


if __name__ == "__main__":
    print("Starting ArXiv Classifier Gradio Demo...")
    print("API URL:", API_BASE_URL)
    print("\nMake sure the Django server is running:")
    print("  cd backend && python manage.py runserver\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
