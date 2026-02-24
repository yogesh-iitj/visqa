"""
ViSQA Gradio Demo — Interactive web interface for video segmentation.

Launch:
    python scripts/demo.py
    python scripts/demo.py --port 7860 --share
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_demo():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio>=4.0.0")

    from visqa import ViSQAPipeline

    # Global pipeline (loaded once)
    _pipeline = None

    def get_pipeline(grounder_type: str, sam2_size: str):
        nonlocal _pipeline
        size_map = {
            "tiny": "sam2_hiera_tiny.yaml",
            "small": "sam2_hiera_small.yaml",
            "base+": "sam2_hiera_base_plus.yaml",
            "large": "sam2_hiera_large.yaml",
        }
        _pipeline = ViSQAPipeline(
            grounder_type=grounder_type,
            sam2_model_cfg=size_map[sam2_size],
        )
        return f"✅ Pipeline loaded: {grounder_type} + SAM2-{sam2_size}"

    def run_inference(
        video_file,
        queries_text: str,
        grounder_type: str,
        sam2_size: str,
        frame_stride: int,
        box_threshold: float,
    ):
        if video_file is None:
            return None, "❌ Please upload a video."

        queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
        if not queries:
            return None, "❌ Please enter at least one query."

        try:
            pipeline = ViSQAPipeline(
                grounder_type=grounder_type,
                sam2_model_cfg={
                    "tiny": "sam2_hiera_tiny.yaml",
                    "small": "sam2_hiera_small.yaml",
                    "base+": "sam2_hiera_base_plus.yaml",
                    "large": "sam2_hiera_large.yaml",
                }[sam2_size],
                frame_stride=frame_stride,
                box_threshold=box_threshold,
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                result = pipeline.run(
                    video_path=video_file,
                    queries=queries,
                    output_dir=tmp_dir,
                    save_video=True,
                )

                summary = "## Results\n\n"
                for qr in result.results:
                    valid = (qr.masks.sum(axis=(1, 2)) > 0).sum()
                    avg_score = qr.scores[qr.scores > 0].mean() if (qr.scores > 0).any() else 0
                    summary += f"**Query:** `{qr.query}`\n"
                    summary += f"- Detected in **{valid}/{len(qr.masks)}** frames\n"
                    summary += f"- Average confidence: **{avg_score:.3f}**\n\n"

                return result.output_video_path, summary

        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"

    with gr.Blocks(title="ViSQA — Video Segmentation & Query Anchoring",
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🎬 ViSQA — Video Segmentation & Query Anchoring
        
        Describe what you want to track in natural language and ViSQA will:
        - 🔍 **Ground** the objects with bounding boxes
        - 🎭 **Segment** them with pixel-level masks
        - 🕰️ **Track** them across the entire video
        """)

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video", sources=["upload"])

                queries_input = gr.Textbox(
                    label="Queries (one per line)",
                    placeholder="the person in the red jacket\nthe dog running\nthe white car",
                    lines=4,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    grounder_choice = gr.Radio(
                        ["gdino", "owlvit", "ensemble"],
                        value="gdino",
                        label="Grounding Model",
                    )
                    sam2_size = gr.Radio(
                        ["tiny", "small", "base+", "large"],
                        value="large",
                        label="SAM2 Model Size",
                    )
                    frame_stride = gr.Slider(1, 5, value=1, step=1, label="Frame Stride")
                    box_threshold = gr.Slider(0.1, 0.7, value=0.3, step=0.05, label="Box Threshold")

                run_btn = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Segmented Video")
                results_md = gr.Markdown("Results will appear here...")

        run_btn.click(
            fn=run_inference,
            inputs=[video_input, queries_input, grounder_choice, sam2_size,
                    frame_stride, box_threshold],
            outputs=[video_output, results_md],
        )

        gr.Examples(
            examples=[
                ["examples/walking.mp4", "the person walking\nthe bicycle"],
                ["examples/soccer.mp4", "the soccer ball\nthe goalkeeper"],
            ],
            inputs=[video_input, queries_input],
            label="Example Videos",
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch ViSQA Gradio Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
