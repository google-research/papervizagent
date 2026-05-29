# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel Streamlit Demo for PaperVizAgent
Accepts user text input, duplicates it 10 times, and runs parallel processing
to generate multiple diagram candidates for comparison.
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("DEBUG: Importing agents...")
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils import generation_utils
    from utils.paperviz_processor import PaperVizProcessor
    print("DEBUG: Imported utils")

    import yaml
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperVizAgent Parallel Demo",
    page_icon="🍌"
)

ASPECT_RATIO_OPTIONS = ["8:1", "21:9", "16:9", "3:2", "4:3", "1:1", "9:16"]
DEFAULT_ASPECT_RATIO_INDEX = ASPECT_RATIO_OPTIONS.index("16:9")

DEMO_EXAMPLE_PRESETS = {
    "PaperVizAgent Framework (Original)": {
        "method": r"""## Methodology: The PaperVizAgent Framework

In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts.""",
        "caption": (
            "Figure 1: Overview of our PaperVizAgent framework. Given the source context "
            "and communicative intent, we first apply a Linear Planning Phase to retrieve "
            "relevant reference examples and synthesize a stylistically optimized description. "
            "We then use an Iterative Refinement Loop (consisting of Visualizer and Critic "
            "agents) to transform the description into visual output and conduct multi-round "
            "refinements to produce the final academic illustration."
        ),
    },
    "[Wide Test] PaperVizAgent Extra-Wide Pipeline": {
        "method": r"""## Methodology: The PaperVizAgent Framework

In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. PaperVizAgent orchestrates five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams.

**Visual layout requirement:** Design this figure as an extra-wide panoramic pipeline. Use a strict left-to-right horizontal flow. Place inputs on the far left, agent stages across the center from left to right, and the final illustration on the far right. Keep all five agents visible in one continuous row with clear arrows between stages.

### Retriever Agent
Given source context $S$ and communicative intent $C$, the Retriever selects $N$ relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from a reference set $\mathcal{R}$:
$$\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)$$
It ranks candidates by research domain and diagram type, prioritizing visual structure over topic similarity.

### Planner Agent
The Planner converts method content and intent into a detailed textual illustration description $P$ using in-context learning from retrieved examples:
$$P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})$$

### Stylist Agent
The Stylist refines $P$ using an automatically synthesized aesthetic guideline $\mathcal{G}$ (color palette, shapes, arrows, layout, typography):
$$P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})$$

### Visualizer Agent
The Visualizer renders images from descriptions. At iteration $t$:
$$I_t = \text{Image-Gen}(P_t), \quad P_0 = P^*$$

### Critic Agent
The Critic inspects $I_t$ against $(S, C)$ and returns an improved description:
$$P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$$
The Visualizer–Critic loop runs for $T=3$ rounds, producing final output $I = I_T$.

### Suggested figure structure (left → right)
1. **Inputs:** boxes for Methodology Section ($S$) and Figure Caption ($C$)
2. **Planning phase:** Retriever → Planner → Stylist (three agent boxes in a row)
3. **Generation loop:** Visualizer ↔ Critic (cyclic refinement, 3 rounds)
4. **Output:** Final Publication-Quality Diagram
Use thin orthogonal arrows, white background, soft academic colors, and compact labels under each agent icon.""",
        "caption": """Figure 1: Overview of the PaperVizAgent framework (publication-style architecture diagram).

Create an extra-wide panoramic pipeline figure with a clean white background and NeurIPS-style academic aesthetics.

**Composition (strict left-to-right):**
- Far left: two input blocks — "Methodology Section (S)" and "Figure Caption (C)"
- Center-left to center: linear planning chain — Retriever → Planner → Stylist, each as a labeled rounded rectangle with a small role icon
- Center-right: iterative refinement loop — Visualizer and Critic connected by curved bidirectional arrows, annotated "T=3 rounds"
- Far right: output block — "Final Academic Illustration"

**Style constraints:**
- Horizontal layout only; do not stack agents vertically
- Use consistent arrow thickness, muted blue/gray palette, sans-serif labels
- No figure title inside the image
- Emphasize data flow with directional arrows between every stage
- Leave generous horizontal spacing so the diagram reads clearly at full paper width""",
    },
}


def ensure_valid_aspect_ratio_state(state_key: str):
    """Reset stale Streamlit aspect-ratio widget values (e.g. removed 32:9)."""
    current = st.session_state.get(state_key)
    if current not in ASPECT_RATIO_OPTIONS:
        st.session_state[state_key] = ASPECT_RATIO_OPTIONS[DEFAULT_ASPECT_RATIO_INDEX]


def load_example_preset():
    """Streamlit callback: populate text areas when an example preset is selected."""
    preset_name = st.session_state.get("demo_example_preset", "None")
    if preset_name == "None":
        st.session_state["_loaded_example_preset"] = None
        return
    apply_demo_example_preset(preset_name)


def apply_demo_example_preset(preset_name):
    """Load method and caption from a named demo preset into the input widgets."""
    if preset_name in DEMO_EXAMPLE_PRESETS:
        preset = DEMO_EXAMPLE_PRESETS[preset_name]
        st.session_state["method_content_input"] = preset["method"]
        st.session_state["caption_input"] = preset["caption"]
        st.session_state["_loaded_example_preset"] = preset_name

def get_configured_model_options(options_key):
    """Return model options from config; first item is the default."""
    options = model_config_data.get("defaults", {}).get(options_key, []) or []
    return options or ["CONFIGURE_MODEL_IN_YAML"]


def render_model_settings():
    """Shared model selectors used by generation and refinement."""
    text_model_options = get_configured_model_options("model_options")
    image_model_options = get_configured_model_options("image_model_options")

    model_name = st.selectbox(
        "Text Model",
        text_model_options,
        index=0,
        key="selected_text_model",
        help="Used by Planner, Stylist, Critic, and Retriever. First option in yaml is the default.",
    )
    image_model_name = st.selectbox(
        "Image Model",
        image_model_options,
        index=0,
        key="selected_image_model",
        help="Used by Visualizer and the Refine tab. First option in yaml is the default.",
    )
    return model_name, image_model_name

def clean_text(text):
    """Clean text by removing invalid UTF-8 surrogate characters."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Convert base64 string to PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3):
    """Create multiple copies of the input data for parallel processing."""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": generation_utils.normalize_aspect_ratio(aspect_ratio)
        },
        "max_critic_rounds": max_critic_rounds  # Add critic rounds control
    }
    
    # Create num_copies identical inputs, each with a unique identifier
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    model_name="",
    image_model_name="",
):
    """Process multiple candidates in parallel using PaperVizProcessor."""
    # Create experiment config
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        model_name=model_name,
        image_model_name=image_model_name,
        work_dir=Path(__file__).parent,
    )
    
    # Initialize processor with all agents
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    
    # Process all candidates in parallel (concurrency controlled by processor)
    results = []
    concurrent_num = 10  # Process all 10 in parallel
    
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num, do_eval=False
    ):
        results.append(result_data)
    
    return results

async def refine_image_with_nanoviz(
    image_bytes,
    edit_prompt,
    aspect_ratio="21:9",
    image_size="2K",
    image_model=None,
):
    """
    Refine an image using an Image Editing API.
    """
    from utils import generation_utils

    if not image_model:
        image_options = get_configured_model_options("image_model_options")
        image_model = image_options[0] if image_options else ""
    if not image_model:
        return None, "❌ No image model configured in configs/model_config.yaml"

    if generation_utils.is_openrouter_configured():
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        contents = [
            {"type": "text", "text": edit_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
        ]
        try:
            response_list = await generation_utils.call_openrouter_image_generation_with_retry_async(
                model_name=image_model,
                contents=contents,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                max_attempts=3,
                retry_delay=10,
                error_context="image refinement",
            )
            if response_list and response_list[0] and response_list[0] != "Error":
                return base64.b64decode(response_list[0]), "✅ Image refined successfully!"
            return None, "❌ No image data found in OpenRouter response"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    try:
        from google import genai
        from google.genai import types

        project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
        location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
        google_api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")

        if google_api_key:
            client = genai.Client(api_key=google_api_key)
        else:
            client = genai.Client(vertexai=True, project=project_id, location=location)

        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(
                mime_type="image/jpeg",
                data=image_bytes
            )
        ]

        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=8192,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=image_model,
            contents=contents,
            config=config
        )

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    edited_image_data = part.inline_data.data

                    if isinstance(edited_image_data, bytes):
                        return edited_image_data, "✅ Image refined successfully!"
                    elif isinstance(edited_image_data, str):
                        return base64.b64decode(edited_image_data), "✅ Image refined successfully!"

        return None, "❌ No image data found in response"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_evolution_stages(result, exp_mode):
    """Extract all evolution stages (images and descriptions) from the result."""
    task_name = "diagram"
    stages = []
    
    # Stage 1: Planner output
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 Planner",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "Initial diagram plan based on method content"
        })
    
    # Stage 2: Stylist output (only for demo_full)
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ Stylist",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "Stylistically refined description"
            })
    
    # Stage 3+: Critic iterations
    for round_idx in range(4):  # Check up to 4 rounds
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"
        
        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 Critic Round {round_idx}",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"Refined after critic feedback (iteration {round_idx})"
            })
    
    return stages

def display_candidate_result(result, candidate_id, exp_mode):
    """Display a single candidate result."""
    task_name = "diagram"
    
    # Determine which image to show based on exp_mode
    # For demo modes, always try to find the last critic round
    final_image_key = None
    final_desc_key = None
    
    # Try to find the last critic round
    for round_idx in range(3, -1, -1):  # Check rounds 3, 2, 1, 0
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break
    
    # Fallback if no critic rounds completed
    if not final_image_key:
        if exp_mode == "demo_full":
            # demo_full uses stylist before visualizer
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            # demo_planner_critic uses planner output
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"
    
    # Display the final image
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, width='stretch', caption=f"Candidate {candidate_id} (Final)")
            
            # Add download button
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ Download",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                width='stretch'
            )
        else:
            st.error(f"Failed to decode image for Candidate {candidate_id}")
    else:
        st.warning(f"No image generated for Candidate {candidate_id}")
    
    # Show evolution timeline in an expander
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 View Evolution Timeline ({len(stages)} stages)", expanded=False):
            st.caption("See how the diagram evolved through different pipeline stages")
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Display the image for this stage
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, width='stretch')
                
                # Show description
                if stage['desc_key'] in result:
                    with st.expander(f"📝 Description", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Show critic suggestions if available
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 Critic Suggestions", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ No changes needed - iteration stopped.")
                        else:
                            st.write(cleaned_sugg)
                
                # Add separator between stages (except for the last one)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # If only one stage, show description in simpler expander
        with st.expander(f"📝 View Description", expanded=False):
            if final_desc_key and final_desc_key in result:
                # Clean the text to remove invalid UTF-8 characters
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("No description available")

def main():
    st.title("🍌 PaperVizAgent Demo")
    st.markdown("AI-powered scientific diagram generation and refinement")

    with st.sidebar:
        st.title("🤖 Models")
        model_name, image_model_name = render_model_settings()
        st.divider()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Generate Candidates", "✨ Refine Image"])
    
    # ==================== TAB 1: Generate Candidates ====================
    with tab1:
        st.markdown("### Generate multiple diagram candidates from your method section and caption")
        
        # Sidebar configuration for Tab 1
        with st.sidebar:
            st.title("⚙️ Generation Settings")
            
            exp_mode = st.selectbox(
                "Pipeline Mode",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="Select which agent pipeline to use"
            )
            
            mode_info = {
                "demo_planner_critic": "Planner → Visualizer → Critic → Visualizer",
                "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer. (The stylist can make the diagram more aesthetically pleasing, but prone to be overly simplied. So we recommend trying both modes and select the best one)"
            }
            st.info(f"**Pipeline:** {mode_info[exp_mode]}")
            
            retrieval_setting = st.selectbox(
                "Retrieval Setting",
                ["none", "auto", "manual", "random"],
                index=0,
                key="tab1_retrieval_setting",
                help="How to retrieve reference diagrams: none (no retrieval, recommended without dataset), auto, manual, or random"
            )
            
            num_candidates = st.number_input(
                "Number of Candidates",
                min_value=1,
                max_value=20,
                value=10,
                key="tab1_num_candidates",
                help="How many parallel candidates to generate"
            )
            
            ensure_valid_aspect_ratio_state("tab1_aspect_ratio")
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ASPECT_RATIO_OPTIONS,
                index=DEFAULT_ASPECT_RATIO_INDEX,
                key="tab1_aspect_ratio",
                help="8:1 extra wide, 21:9 ultrawide, 16:9 widescreen, 3:2 classic, 1:1 square (Gemini-supported ratios)",
            )
            
            max_critic_rounds = st.number_input(
                "Max Critic Rounds",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="Maximum number of critic refinement iterations"
            )
        
        st.divider()
        
        # Input section
        st.markdown("## 📝 Input")

        example_preset = st.selectbox(
            "Load Example",
            ["None", *DEMO_EXAMPLE_PRESETS.keys()],
            key="demo_example_preset",
            on_change=load_example_preset,
            help="Loads method text and caption only. Pick aspect ratio separately in the sidebar.",
        )
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            method_content = st.text_area(
                "Method Section Content (Markdown recommended)",
                height=250,
                placeholder="Paste the method section content here...",
                help="The method section from the paper that describes the approach. Markdown format is recommended.",
                key="method_content_input",
            )
        
        with col_input2:
            caption = st.text_area(
                "Figure Caption (Markdown recommended)",
                height=250,
                placeholder="Enter the figure caption...",
                help="The caption or description of the figure to generate. Markdown format is recommended.",
                key="caption_input",
            )
        
        # Process button
        if st.button("🚀 Generate Candidates", type="primary", width='stretch'):
            if not method_content or not caption:
                st.error("Please provide both method content and caption!")
            else:
                with st.spinner(f"Generating {num_candidates} candidates in parallel... This may take a few minutes."):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list, 
                            exp_mode=exp_mode, 
                            retrieval_setting=retrieval_setting,
                            model_name=model_name,
                            image_model_name=image_model_name,
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        # Save results to JSON file
                        try:
                            # Create results directory if it doesn't exist
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate filename with timestamp
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            # Save to JSON with proper encoding handling (like main.py)
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # Clean invalid UTF-8 characters
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)
                            
                            st.session_state["json_file"] = str(json_filename)
                            st.success(f"✅ Successfully generated {len(results)} candidates!")
                            st.info(f"💾 Results saved to: `{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"⚠️ Generated {len(results)} candidates, but failed to save JSON: {e}")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            
            st.divider()
            st.markdown("## 🎨 Generated Candidates")
            st.caption(f"Generated at: {timestamp} | Pipeline: {mode_info.get(current_mode, current_mode)}")
            
            # Show JSON file download if available
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 Results saved to: `{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            width='stretch'
                        )
            
            # Display results in a grid (3 columns)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Add ZIP download button
            st.divider()
            st.markdown("### 💾 Batch Download")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"
                    
                    for candidate_id, result in enumerate(results):
                        
                        # Find the final image key (same logic as display)
                        final_image_key = None
                        
                        # Try to find the last critic round
                        for round_idx in range(3, -1, -1):
                            image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                            if image_key in result and result[image_key]:
                                final_image_key = image_key
                                break
                        
                        # Fallback if no critic rounds completed
                        if not final_image_key:
                            if current_mode == "demo_full":
                                final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                            else:
                                final_image_key = f"target_{task_name}_desc0_base64_jpg"
                        
                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width='stretch'
                )
                st.success("ZIP file ready for download!")
            except Exception as e:
                st.error(f"Failed to create ZIP: {e}")
    
    # ==================== TAB 2: Refine Image ====================
    with tab2:
        st.markdown("### Refine and upscale your diagram to high resolution (2K/4K)")
        st.caption("Upload an image from the candidates or any diagram, describe changes, and generate a high-res version")
        
        # Sidebar for refinement settings
        with st.sidebar:
            st.title("✨ Refinement Settings")
            
            refine_resolution = st.selectbox(
                "Target Resolution",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="Higher resolution takes longer but produces better quality"
            )
            
            ensure_valid_aspect_ratio_state("refine_aspect_ratio")
            refine_aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ASPECT_RATIO_OPTIONS,
                index=DEFAULT_ASPECT_RATIO_INDEX,
                key="refine_aspect_ratio",
                help="8:1 extra wide, 21:9 ultrawide, 16:9 widescreen, 3:2 classic, 1:1 square (Gemini-supported ratios)",
            )
        
        st.divider()
        
        # Upload section
        st.markdown("## 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload the diagram you want to refine"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(uploaded_image, width='stretch')
            
            with col2:
                st.markdown("### Edit Instructions")
                edit_prompt = st.text_area(
                    "Describe the changes you want",
                    height=200,
                    placeholder="E.g., 'Change the color scheme to match academic paper style' or 'Make the text larger and bolder' or 'Keep everything the same but output in higher resolution'",
                    help="Describe what you want to change or use 'Keep everything the same' for just upscaling",
                    key="edit_prompt"
                )
                
                if st.button("✨ Refine Image", type="primary", width='stretch'):
                    if not edit_prompt:
                        st.error("Please provide edit instructions!")
                    else:
                        with st.spinner(f"Refining image to {refine_resolution} resolution... This may take a minute."):
                            try:
                                # Convert PIL image to bytes
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()
                                
                                # Call nanoviz API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution,
                                        image_model=image_model_name,
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Error during refinement: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Display refined result if available
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 Refined Result")
                st.caption(f"Generated at: {st.session_state.get('refine_timestamp', 'N/A')} | Resolution: {refine_resolution}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Before")
                    st.image(uploaded_image, width='stretch')
                
                with col2:
                    st.markdown(f"### After ({refine_resolution})")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, width='stretch')
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {refine_resolution} Image",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        width='stretch'
                    )

if __name__ == "__main__":
    main()
