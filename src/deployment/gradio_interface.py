"""
Gradio interface for healthcare chatbot.
Simple UI with LingBot branding and chat functionality.
"""

from data_processing.data_loader import DataLoader
from pipeline.output_formatter import OutputFormatter
from pipeline.inference_engine import InferenceEngine
from pipeline.query_router import QueryRouter
from pipeline.input_processor import InputProcessor
import gradio as gr
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class LingBotInterface:
    """Gradio interface for LingBot healthcare assistant."""

    def __init__(self):
        """Initialize the LingBot interface."""
        self.input_processor = InputProcessor()
        self.inference_engine = None
        self.output_formatter = OutputFormatter()
        self.query_router = None
        self.data_loaded = False

        # Load data and initialize components
        self._load_data()

    def _load_data(self):
        """Load preprocessed data and initialize components."""
        try:
            print("Loading data and initializing components...")

            # Load data
            data_loader = DataLoader()
            data = data_loader.load_all_data()

            # Initialize query router
            self.query_router = QueryRouter(
                vector_indexer=data['vector_indexer'],
                hospital_data=data['processed_data']['hospital']
            )

            # Initialize inference engine
            self.inference_engine = InferenceEngine()

            self.data_loaded = True
            print("Data loaded successfully!")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_loaded = False

    def process_query(self, user_input: str) -> Tuple[str, str]:
        """
        Process user query and generate response.

        Args:
            user_input: User's input query

        Returns:
            Tuple of (response, status)
        """
        if not self.data_loaded:
            return "Sorry, the system is not ready. Please try again later.", "Error"

        if not user_input or not user_input.strip():
            return "Please enter a question.", "Warning"

        try:
            # Process input
            processed_input = self.input_processor.preprocess_query(user_input)

            if not processed_input['is_valid']:
                return processed_input['error_message'], "Error"

            # Route query
            routing_result = self.query_router.route_query(
                processed_input['cleaned_query'])

            # Generate response based on intent
            if routing_result['intent'] == "Greeting or Chit-chat":
                response = self.output_formatter.format_greeting_response(
                    user_input)
            else:
                # Generate response using inference engine
                response = self.inference_engine.generate_final_response(
                    query=processed_input['cleaned_query'],
                    context=routing_result['context'],
                    intent=routing_result['intent']
                )

            # Format final output
            formatted_response = self.output_formatter.format_response(
                response)

            return formatted_response, "Success"

        except Exception as e:
            error_response = self.output_formatter.format_error_response(
                'generation_error', user_input)
            return error_response, "Error"

    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""

        # Custom CSS for styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo {
            width: 80px;
            height: 80px;
            margin: 0 auto;
            display: block;
        }
        .bot-name {
            font-size: 28px;
            font-weight: bold;
            color: #2c5aa0;
            margin: 10px 0;
        }
        .subtitle {
            font-size: 16px;
            color: #666;
            margin-bottom: 20px;
        }
        .disclaimer {
            font-size: 12px;
            color: #888;
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        """

        # Create interface
        with gr.Blocks(css=css, title="LingBot - Healthcare Assistant") as interface:

            # Header with logo and bot name
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="header">
                        <div class="logo">🤖</div>
                        <div class="bot-name">LingBot</div>
                        <div class="subtitle">Your Healthcare Information Assistant</div>
                    </div>
                    """)

            # Main chat interface
            with gr.Row():
                with gr.Column(scale=4):
                    # Input text box
                    user_input = gr.Textbox(
                        label="Ask me anything about health, lifestyle tips, or hospital information:",
                        placeholder="e.g., What are the symptoms of flu? or How to maintain a healthy diet?",
                        lines=3,
                        max_lines=5
                    )

                    # Submit button
                    submit_btn = gr.Button(
                        "Ask LingBot", variant="primary", size="lg")

            # Response area
            with gr.Row():
                with gr.Column(scale=4):
                    response_output = gr.Textbox(
                        label="LingBot's Response:",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        show_copy_button=True
                    )

            # Status indicator
            with gr.Row():
                with gr.Column(scale=4):
                    status_output = gr.Textbox(
                        label="Status:",
                        lines=1,
                        interactive=False,
                        visible=False
                    )

            # Disclaimer
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""
                    <div class="disclaimer">
                        <strong>Disclaimer:</strong> This is for educational purposes only and not a substitute for professional medical advice. 
                        Always consult a healthcare professional for personalized medical guidance.
                    </div>
                    """)

            # Example questions
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""
                    <div style="margin-top: 20px;">
                        <h4>Example Questions:</h4>
                        <ul style="text-align: left; color: #666;">
                            <li>What are the symptoms of diabetes?</li>
                            <li>How much sleep should I get?</li>
                            <li>What are the hospital's working hours?</li>
                            <li>Give me tips for healthy eating</li>
                            <li>How to book an appointment?</li>
                        </ul>
                    </div>
                    """)

            # Event handlers
            def handle_submit(query):
                if not query.strip():
                    return "", "Please enter a question.", gr.update(visible=True)

                response, status = self.process_query(query)
                return response, status, gr.update(visible=True)

            def handle_enter(query):
                return handle_submit(query)

            # Connect events
            submit_btn.click(
                fn=handle_submit,
                inputs=[user_input],
                outputs=[response_output, status_output,
                         gr.update(visible=True)]
            )

            user_input.submit(
                fn=handle_enter,
                inputs=[user_input],
                outputs=[response_output, status_output,
                         gr.update(visible=True)]
            )

        return interface

    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch the Gradio interface."""
        interface = self.create_interface()

        print("Starting LingBot interface...")
        print(
            f"Server will be available at: http://{server_name}:{server_port}")

        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the LingBot interface."""
    try:
        lingbot = LingBotInterface()
        lingbot.launch()
    except Exception as e:
        print(f"Error starting LingBot: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

