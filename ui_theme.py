import gradio as gr

def get_theme():
    # प्रोफेशनल डार्क ऑरेंज थीम
    return gr.themes.Default(
        primary_hue="orange",
        secondary_hue="stone",
        neutral_hue="gray",
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        block_title_text_weight="700",
    )
  
