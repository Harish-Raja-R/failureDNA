"""FailureDNA: Hugging Face Spaces Interactive Research Application.

Compatible with Free Hugging Face Spaces tier (sdk: gradio).
Reuses frozen Pre-LN Temporal Transformer, RegimeAwarePreprocessor,
and Spherical K-Means signature clusterer.
"""

import os
from pathlib import Path
import sys

import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from failuredna.inference.service import get_inference_service

# Initialize singleton inference service
service = get_inference_service()

DEMO_ENGINES = {
    "Unit #001 (Nominal Initial Cycles)": 1,
    "Unit #034 (Transition Degradation)": 34,
    "Unit #098 (Critical Pre-Failure)": 98,
}


try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn


@gpu_decorator
def analyze_telemetry_csv(file_obj):
    """Process uploaded CSV telemetry file."""
    if file_obj is None:
        return "Please upload a CSV or TXT file.", None, None, None

    try:
        result = service.analyze_csv_file(file_obj.name)

        rul = result.get("predicted_rul", 0.0)
        unclipped = result.get("predicted_rul_unclipped", 0.0)
        sig_name = result.get("signature_name", "Unknown")
        sig_desc = result.get("signature_description", "")
        dist = result.get("distance_to_centroid")
        dist_str = f" // Centroid Distance: {dist:.3f}" if dist is not None else ""

        rul_text = f"### ⏱️ Predicted Remaining Useful Life: **{rul:.1f} Cycles**\n\n*(Unclipped Continuous RUL: {unclipped:.1f} cycles)*"
        sig_text = f"### 🧬 Discovered Signature: **{sig_name}**\n\n{sig_desc}{dist_str}"
        warnings_text = "\n".join(f"- {w}" for w in result.get("warnings", [])) if result.get("warnings") else "None. Sequence satisfies all constraints."

        # Sensor summary
        sensor_summary = pd.DataFrame([
            {
                "Sensor": s,
                "Start": round(t["start"], 2),
                "End": round(t["end"], 2),
                "Min": round(t["min"], 2),
                "Max": round(t["max"], 2),
                "Mean": round(t["mean"], 2),
                "Delta": round(t["delta"], 2),
            }
            for s, t in list(result.get("sensor_trends", {}).items())[:8]
        ])

        return rul_text, sig_text, warnings_text, sensor_summary
    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None, None


@gpu_decorator
def analyze_demo_engine(engine_name):
    """Run analysis on selected demo engine."""
    unit_id = DEMO_ENGINES.get(engine_name, 1)
    test_path = Path("data/raw/test_FD004.txt")

    if not test_path.exists():
        return f"Dataset file {test_path} not found.", None, None, None

    try:
        from failuredna.data.loader import ID_COL, load_raw_cmapss_file
        raw_test = load_raw_cmapss_file(test_path)
        engine_df = raw_test[raw_test[ID_COL] == unit_id].copy()

        result = service.analyze_sequence(unit_id=unit_id, sequence_records=engine_df.to_dict(orient="records"))

        rul = result.get("predicted_rul", 0.0)
        unclipped = result.get("predicted_rul_unclipped", 0.0)
        sig_name = result.get("signature_name", "Unknown")
        sig_desc = result.get("signature_description", "")
        dist = result.get("distance_to_centroid")
        dist_str = f" // Centroid Distance: {dist:.3f}" if dist is not None else ""

        rul_text = f"### ⏱️ Predicted Remaining Useful Life: **{rul:.1f} Cycles**\n\n*(Unclipped RUL: {unclipped:.1f} cycles)*"
        sig_text = f"### 🧬 Latent Signature: **{sig_name}**\n\n{sig_desc}{dist_str}"
        warnings_text = "\n".join(f"- {w}" for w in result.get("warnings", [])) if result.get("warnings") else "No warnings. All temporal windows validated."

        sensor_summary = pd.DataFrame([
            {
                "Sensor": s,
                "Start": round(t["start"], 2),
                "End": round(t["end"], 2),
                "Min": round(t["min"], 2),
                "Max": round(t["max"], 2),
                "Mean": round(t["mean"], 2),
                "Delta": round(t["delta"], 2),
            }
            for s, t in list(result.get("sensor_trends", {}).items())[:8]
        ])

        return rul_text, sig_text, warnings_text, sensor_summary
    except Exception as e:
        return f"Error analyzing engine: {str(e)}", None, None, None


# Gradio UI Theme & Layout
custom_theme = gr.themes.Soft(
    primary_hue="pink",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui"],
)

with gr.Blocks(theme=custom_theme, title="FailureDNA: Industrial AI Platform") as demo:
    gr.Markdown(
        """
        # 🧬 FailureDNA: Industrial AI Research Platform
        ### Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series
        **NASA C-MAPSS FD004 Test Fleet Evaluation // Zero-Leakage Preprocessing // Temporal Transformer Encoder**
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔬 Live Engine Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Select Engine or Upload Telemetry")
                    engine_dropdown = gr.Dropdown(
                        choices=list(DEMO_ENGINES.keys()),
                        value="Unit #098 (Critical Pre-Failure)",
                        label="Pre-Configured Test Trajectories",
                    )
                    run_demo_btn = gr.Button("Analyze Selected Engine", variant="primary")

                    gr.Markdown("---")
                    file_input = gr.File(label="Upload Custom Engine Telemetry (CSV)", file_types=[".csv", ".txt"])
                    run_file_btn = gr.Button("Analyze Uploaded File")

                with gr.Column(scale=2):
                    gr.Markdown("### 2. Prognostic & Diagnostic Results")
                    rul_output = gr.Markdown(value="Select an engine or upload telemetry to begin analysis.")
                    sig_output = gr.Markdown()
                    warnings_output = gr.Markdown()
                    gr.Markdown("#### Sensor Drift Summary")
                    trends_table = gr.Dataframe(label="Key Sensor Channels (First 8)")

            run_demo_btn.click(
                fn=analyze_demo_engine,
                inputs=[engine_dropdown],
                outputs=[rul_output, sig_output, warnings_output, trends_table],
            )
            run_file_btn.click(
                fn=analyze_telemetry_csv,
                inputs=[file_input],
                outputs=[rul_output, sig_output, warnings_output, trends_table],
            )

        with gr.TabItem("📊 Official Benchmark"):
            gr.Markdown(
                """
                ### NASA C-MAPSS FD004 Official 248-Engine Benchmark Evaluation
                - **Primary Metric**: MAE (Cycles), RMSE (Cycles), NASA Asymmetric Penalty Score, R² Coefficient.
                - **Hypothesis Testing**: Paired bootstrap ($B=10,000$, seed 42) with omnibus Holm-Bonferroni FWER control.
                - **Scientific Verdict**: Under the Holm correction, XGBoost maintains a statistically significant MAE advantage over FailureDNA ($\Delta = +2.86$ cycles, $p = 0.0150$). RMSE and NASA differences are non-significant. FailureDNA does not establish prognostic superiority over gradient-boosted trees on FD004.
                """
            )

        with gr.TabItem("📖 Research Specifications"):
            gr.Markdown(
                """
                ### Architectural Formulation
                - **Encoder Backbone**: 4-Layer Pre-LayerNorm Temporal Transformer ($d_{model}=128, H=8, d_{ff}=256$)
                - **Contrastive Objective**: InfoNCE / NT-Xent with cosine similarity and $\\tau = 0.07$
                - **Context Window**: $W = 50$ cycles with masked mean temporal pooling
                - **Signature Discovery**: Spherical K-Means ($K=4$) on normalized hypersphere
                - **Downstream RUL Estimator**: Linear Ridge Regression ($\\alpha = 10.0$) on frozen embeddings
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
