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
from failuredna.inference.service import FailureDNAInferenceService

# Initialize singleton inference service
service = FailureDNAInferenceService()

DEMO_ENGINES = {
    "Unit #001 (Nominal Initial Cycles)": 1,
    "Unit #034 (Transition Degradation)": 34,
    "Unit #098 (Critical Pre-Failure)": 98,
}


def analyze_telemetry_csv(file_obj):
    """Process uploaded CSV telemetry file."""
    if file_obj is None:
        return "Please upload a CSV file.", None, None, None

    try:
        df = pd.read_csv(file_obj.name)
        result = service.analyze_csv_file(file_obj.name)

        rul_text = f"**Predicted RUL:** {result.predicted_rul:.1f} cycles (Cap: 130 cycles)"
        sig_text = f"**Discovered Signature:** {result.signature_name} ({result.signature_description})"
        warnings_text = "\n".join(f"- {w}" for w in result.warnings) if result.warnings else "None. Sequence satisfies all constraints."

        # Sensor summary
        sensor_summary = pd.DataFrame([
            {
                "Sensor": s,
                "Start": t.start,
                "End": t.end,
                "Min": t.min,
                "Max": t.max,
                "Mean": t.mean,
                "Delta": t.delta,
            }
            for s, t in list(result.sensor_trends.items())[:6]
        ])

        return rul_text, sig_text, warnings_text, sensor_summary
    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None, None


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

        rul_text = f"### ⏱️ Predicted Remaining Useful Life: **{result.predicted_rul:.1f} Cycles**\n\n*(Unclipped RUL: {result.predicted_rul_unclipped:.1f} cycles)*"
        sig_text = f"### 🧬 Latent Signature: **{result.signature_name}**\n\n{result.signature_description}\n\n*Centroid Distance: {result.distance_to_centroid:.3f}*"
        warnings_text = "\n".join(f"- {w}" for w in result.warnings) if result.warnings else "No warnings. All temporal windows validated."

        sensor_summary = pd.DataFrame([
            {
                "Sensor": s,
                "Start": round(t.start, 2),
                "End": round(t.end, 2),
                "Min": round(t.min, 2),
                "Max": round(t.max, 2),
                "Mean": round(t.mean, 2),
                "Delta": round(t.delta, 2),
            }
            for s, t in list(result.sensor_trends.items())[:8]
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
