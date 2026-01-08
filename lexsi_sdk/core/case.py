from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import plotly.graph_objects as go
import pandas as pd
from IPython.display import SVG, display
from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.xai_uris import EXPLAINABILITY_SUMMARY, GET_TRIGGERS_DAYS_URI
import base64
from PIL import Image


class Case(BaseModel):
    """Represents an explainability case for a prediction. Provides visualization helpers such as SHAP, LIME, Integrated Gradients, and decision paths."""

    status: str
    true_value: str | int
    pred_value: str | int
    pred_category: str | int
    observations: List
    shap_feature_importance: Optional[Dict] = {}
    lime_feature_importance: Optional[Dict] = {}
    ig_features_importance: Optional[Dict] = {}
    dlb_feature_importance: Optional[Dict] = {}
    similar_cases: List
    is_automl_prediction: Optional[bool] = False
    model_name: str
    case_prediction_path: Optional[str] = ""
    case_prediction_svg: Optional[str] = ""
    observation_checklist: Optional[List] = []
    policy_checklist: Optional[List] = []
    final_decision: Optional[str] = ""
    unique_identifier: Optional[str] = ""
    tag: Optional[str] = ""
    created_at: Optional[str] = ""
    data: Optional[Dict] = {}
    similar_cases_data: Optional[List] = []
    audit_trail: Optional[dict] = {}
    project_name: Optional[str] = ""
    image_data: Optional[Dict] = {}
    data_id: Optional[str] = ""
    summary: Optional[str] = ""
    model_config = ConfigDict(protected_namespaces=())

    api_client: APIClient

    def __init__(self, **kwargs):
        """Capture API client used to fetch additional explainability data.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def explainability_shap_feature_importance(self):
        """Plot a horizontal bar chart showing SHAP-based feature importance for the case. Uses stored Shapley values for features."""
        fig = go.Figure()

        if len(list(self.shap_feature_importance.values())) < 1:
            return "No Shap Feature Importance for the case"

        if isinstance(list(self.shap_feature_importance.values())[0], dict):
            for col in self.shap_feature_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.shap_feature_importance[col].values()),
                        y=list(self.shap_feature_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.shap_feature_importance.values()),
                    y=list(self.shap_feature_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_ig_feature_importance(self):
        """Plot a horizontal bar chart showing Integrated Gradients-based feature importance for the case."""
        fig = go.Figure()

        if len(list(self.ig_features_importance.values())) < 1:
            return "No IG Feature Importance for the case"

        if isinstance(list(self.ig_features_importance.values())[0], dict):
            for col in self.ig_features_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.ig_features_importance[col].values()),
                        y=list(self.ig_features_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.ig_features_importance.values()),
                    y=list(self.ig_features_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_lime_feature_importance(self):
        """Plot a horizontal bar chart showing LIME-based feature importance for the case."""
        fig = go.Figure()

        if len(list(self.lime_feature_importance.values())) < 1:
            return "No Lime Feature Importance for the case"

        if isinstance(list(self.lime_feature_importance.values())[0], dict):
            for col in self.lime_feature_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.lime_feature_importance[col].values()),
                        y=list(self.lime_feature_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.lime_feature_importance.values()),
                    y=list(self.lime_feature_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_dlb_feature_importance(self):
        """Plot a horizontal bar chart showing Deep Lift Bayesian (DLB)-based feature importance for the case."""
        fig = go.Figure()
        if len(list(self.dlb_feature_importance.values())) < 1:
            return "No DLB Feature Importance for the case"

        if isinstance(list(self.dlb_feature_importance.values())[0], dict):
            for col in self.dlb_feature_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.dlb_feature_importance[col].values()),
                        y=list(self.dlb_feature_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.dlb_feature_importance.values()),
                    y=list(self.dlb_feature_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_prediction_path(self):
        """Display the model’s prediction path as a sequence of decision nodes for the case, typically visualized as an SVG or plot."""
        svg = SVG(self.case_prediction_svg)
        display(svg)

    def explainability_raw_data(self) -> pd.DataFrame:
        """Return the raw data used for the case as a DataFrame, with feature names and values.

        :return: raw data dataframe
        """
        raw_data_df = (
            pd.DataFrame([self.data])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
        )
        return raw_data_df

    def explainability_observations(self) -> pd.DataFrame:
        """Return a DataFrame listing the checklist of observations (e.g., heuristics or warnings) associated with the case.

        :return: observations dataframe
        """
        observations_df = pd.DataFrame(self.observation_checklist)

        return observations_df

    def explainability_policies(self) -> pd.DataFrame:
        """Return a DataFrame listing policies or rules applied during the model’s decision for the case.

        :return: policies dataframe
        """
        policy_df = pd.DataFrame(self.policy_checklist)

        return policy_df

    def explainability_decision(self) -> pd.DataFrame:
        """Return a DataFrame summarizing the final decision for the case, including the true value, predicted value, predicted category, and final decision.

        :return: decision dataframe
        """
        data = {
            "True Value": self.true_value,
            "Prediction Value": self.pred_value,
            "Prediction Category": self.pred_category,
            "Final Prediction": self.final_decision,
        }
        decision_df = pd.DataFrame([data])

        return decision_df

    def explainability_similar_cases(self) -> pd.DataFrame | str:
        """Return a DataFrame of cases similar to the current case (if similar cases are available). If no similar cases are found, returns a message.

        :return: similar cases dataframe
        """
        if not self.similar_cases_data:
            return "No similar cases found. Or add 'similar_cases' in components case_info()"

        similar_cases_df = pd.DataFrame(self.similar_cases_data)
        return similar_cases_df

    def explainability_gradcam(self):
        """Visualize Grad-CAM results for image data, showing heatmaps and superimposed regions that contributed to the prediction."""
        if not self.image_data.get("gradcam", None):
            return "No Gradcam method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("heatmap"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("superimposed"),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Superimposed",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_shap(self):
        """Render a SHAP attribution plot for image cases, displaying attributions as an overlay on the original image."""
        if not self.image_data.get("shap", None):
            return "No Shap method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("shap", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_lime(self):
        """Render a LIME attribution plot for image cases, displaying attributions as an overlay on the original image."""
        if not self.image_data.get("lime", None):
            return "No Lime method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("lime", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_integrated_gradients(self):
        """Render an integrated gradients attribution plot for image cases, showing positive and negative attributions side-by-side."""
        if not self.image_data.get("integrated_gradients", None):
            return "No Integrated Gradients method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "attributions"
                ),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "positive_attributions"
                ),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "negative_attributions"
                ),
                xref="x",
                yref="y",
                x=2.4,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Positive Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=2.9,
            y=0.1,
            text="Negative Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def alerts_trail(self, page_num: Optional[int] = 1, days: Optional[int] = 7):
        """Fetch alerts for this case over the given window.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        if days == 7:
            return pd.DataFrame(self.audit_trail.get("alerts", {}))
        resp = self.api_client.post(
            f"{GET_TRIGGERS_DAYS_URI}?project_name={self.project_name}&page_num={page_num}&days={days}"
        )
        if resp.get("details"):
            return pd.DataFrame(resp.get("details"))
        else:
            return "No alerts found."

    def audit(self):
        """Return stored audit trail.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.audit_trail

    def feature_importance(self, feature: str):
        """Return feature importance values for a specific feature.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        if self.shap_feature_importance:
            return self.shap_feature_importance.get(feature, {})
        elif self.lime_feature_importance:
            return self.lime_feature_importance.get(feature, {})
        elif self.ig_features_importance:
            return self.ig_features_importance.get(feature, {})
        else:
            return "No Feature Importance found for the case"

    def explainability_summary(self):
        """Request or return cached explainability summary text.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        if self.data_id and not self.summary:
            payload = {
                "project_name": self.project_name,
                "viewed_case_id": self.data_id,
            }
            res = self.api_client.post(EXPLAINABILITY_SUMMARY, payload)
            if not res.get("success"):
                raise Exception(res.get("details", "Failed to summarize"))

            self.summary = res.get("details")
            return res.get("details")

        return self.summary


class CaseText(BaseModel):
    """Explainability view for text-based cases. Supports token-level importance, attention visualization, and LLM output analysis."""

    model_name: str
    status: str
    prompt: str
    output: str
    explainability: Optional[Dict] = {}
    audit_trail: Optional[Dict] = {}

    def prompt(self):
        """Get prompt
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.prompt

    def output(self):
        """Get output
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.output

    def explainability_raw_data(self) -> pd.DataFrame:
        """Return the raw data used for the case as a DataFrame, with feature names and values.

        :return: raw data dataframe
        """
        raw_data_df = (
            pd.DataFrame([self.explainability.get("feature_importance", {})])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
            .sort_values(by="Value", ascending=False)
        )
        return raw_data_df

    def explainability_feature_importance(self):
        """Plots Feature Importance chart
        Encapsulates a small unit of SDK logic and returns the computed result."""
        fig = go.Figure()
        feature_importance = self.explainability.get("feature_importance", {})

        if not feature_importance:
            return "No Feature Importance for the case"
        raw_data_df = (
            pd.DataFrame([feature_importance])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
            .sort_values(by="Value", ascending=False)
        )
        fig.add_trace(
            go.Bar(x=raw_data_df["Value"], y=raw_data_df["Feature"], orientation="h")
        )
        fig.update_layout(
            barmode="relative",
            height=max(400, len(raw_data_df) * 20),
            width=800,
            yaxis=dict(
                autorange="reversed",
                tickmode="array",
                tickvals=list(raw_data_df["Feature"]),
                ticktext=list(raw_data_df["Feature"]),
                tickfont=dict(size=10),
            ),
            bargap=0.01,
            margin=dict(l=150, r=20, t=30, b=30),
            legend_orientation="h",
            legend_x=0.1,
            legend_y=0.5,
        )

        fig.show(config={"displaylogo": False})

    def network_graph(self):
        """Decode and return a base64-encoded network graph image.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        network_graph_data = self.explainability.get("network_graph", {})
        if not network_graph_data:
            return "No Network graph found for this case"
        base64_str = network_graph_data
        try:
            img_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(img_bytes))
            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def token_attribution_graph(self):
        """Decode and return a base64-encoded token attribution graph.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        relevance_data = self.explainability.get("relevance", {})
        if not relevance_data:
            return "No Token Attribution graph found for this case"
        base64_str = relevance_data
        try:
            img_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(img_bytes))
            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def audit(self):
        """Return audit details for the text case.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.audit_trail
