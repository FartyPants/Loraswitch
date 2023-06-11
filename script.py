from pathlib import Path
import gradio as gr
from modules import utils
from modules import shared
from modules.models import unload_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import math
import os

params = {
    "display_name": "Lora",
    "is_tab": False,
}

refresh_symbol = '\U0001f504'  # 🔄

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component])
    return refresh_button

def get_available_models():

    prior_set = []
    if hasattr(shared.model,'peft_config'):
        for adapter_name in shared.model.peft_config.items():
            print(f"Found adapters: {adapter_name[0]}")
            prior_set.append(adapter_name[0])

        model_name = None
        if hasattr(shared.model, 'active_adapter'):
            model_name = shared.model.active_adapter
        print(f"Current Active Adapter: {model_name}")

    return prior_set

def changemenu(item):
      
    prior_set = list(shared.lora_names)
    if hasattr(shared.model, 'set_adapter') and hasattr(shared.model, 'active_adapter'):
        if prior_set:
            shared.model.set_adapter(item)
            print (f"Current Active Adapter: {shared.model.active_adapter}")   

        


def ui():

    model_name = None
    if hasattr(shared.model, 'active_adapter'):
        model_name = shared.model.active_adapter

    with gr.Accordion("Lora Switcharoo", open=True):
       
        with gr.Row():
               
            with gr.Column():
                with gr.Row():
                    gr_modelmenu = gr.Radio(choices=get_available_models(), value=model_name, label='Loaded Lora(s)', interactive=True)
                    create_refresh_button(gr_modelmenu, lambda: None, lambda: {'choices': get_available_models(),'value': shared.model.active_adapter}, 'refresh-button')
                    #refresh_button = ToolButton(value=refresh_symbol)

    gr_modelmenu.change(changemenu,gr_modelmenu,None)
   

