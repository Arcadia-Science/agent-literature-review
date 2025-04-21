"""
Main module for running the AI laboratory.

This code was developed with the assistance of Claude Code.
"""
import argparse
import os
import re
from typing import List, Optional

from src.config import LabConfig
from src.agents import Laboratory
from src.utils.document_loader import load_document


def display_cost_estimate(lab):
    """
    Display current cost estimate for the laboratory session.
    
    Args:
        lab: Laboratory instance to get cost from.
    """
    cost_data = lab.get_cost_estimate()
    total_cost = cost_data["total_cost"]
    
    print("\n\033[1m💰 COST ESTIMATE\033[0m")
    print("═" * 50)
    print(f"Total estimated cost: \033[1;33m${total_cost:.4f}\033[0m")
    
    # Display breakdown by model
    if cost_data["cost_breakdown"]:
        print("\nBreakdown by model:")
        for model, data in cost_data["cost_breakdown"].items():
            input_tokens = data["input_tokens"]
            output_tokens = data["output_tokens"]
            model_cost = data["total_cost"]
            print(f"  \033[1;36m{model}\033[0m: ${model_cost:.4f} ({input_tokens} input + {output_tokens} output tokens)")
    
    print("═" * 50)
    print("\033[3mNote: This is an approximate cost estimate based on current pricing.\033[0m")
    print()


def run_lab(config_path: str) -> None:
    """
    Run the AI laboratory with the given configuration.
    
    Args:
        config_path: Path to the configuration file.
    """
    try:
        config = LabConfig(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {str(e)}")
        print(f"Please create a configuration file at {config_path} using the example in the README.")
        return
    
    
    lab = Laboratory(config)
    print(f"\n\033[1;32m=== Welcome to the AI Laboratory, {config.user_name}! ===\033[0m")
    print(f"You are speaking with {len(lab.agents)} AI agents:")
    
    # Display agents with clear formatting
    print("\n\033[1mYour AI research team:\033[0m")
    for agent in lab.agents:
        print(f"  \033[1;34m{agent.name} ({agent.specialty}):\033[0m {agent.description}")
    
    print("\n\033[1m=== Instructions ===\033[0m")
    print("• Type your message and press Enter to start the discussion")
    print("• To address specific agents directly, use '@AgentName:' at the start of a line")
    print("• Type '/discuss <topic>' to start a focused multi-round agent discussion")
    print("• Type '/search <query> [--type arxiv|biorxiv] [--results <number>] [--months <number>]' to search for information")
    print("  - For bioRxiv searches, use '--months' to specify how many months back to search (default: 12)")
    print("• Type '/read_folder <folder_path>' to have agents read and discuss papers in a local folder")
    print("  - Reads PDF files from the specified folder and has agents discuss them directly")
    print("• Type '/cost' to see estimated API costs of the current session")
    print("• Type 'exit' to end the session")
    
    print("\n\033[1m=== Beginning of Discussion ===\033[0m")
    
    while True:
        try:
            try:
                print(f"\n\033[1;36m{config.user_name}:\033[0m")
                user_input = input("  ")
                
                if user_input.lower() == 'exit':
                    print("Ending session. Goodbye!")
                    # Display final cost estimate
                    display_cost_estimate(lab)
                    break
                    
                if not user_input.strip():  # Skip empty inputs
                    continue
            except EOFError:
                print("\nInput terminated. Ending session.")
                break
                
            # Check for discussion command
            if user_input.lower().startswith('/discuss'):
                # Extract topic (everything after the command)
                topic = user_input[8:].strip()
                if not topic:
                    print("\033[93mPlease specify a topic for discussion: /discuss <topic>\033[0m")
                    continue
                    
                # Facilitate a multi-round discussion between agents
                lab.user_message(f"I'd like you all to discuss this topic: {topic}")
                lab.facilitate_agent_discussion(topic, rounds=2)
                continue
            
            # Check for read_folder command
            if user_input.lower().startswith('/read_folder'):
                # Extract the folder path
                folder_path = user_input[12:].strip()
                
                if not folder_path:
                    print("\033[93mPlease provide a folder path: /read_folder <path>\033[0m")
                    continue
                
                # Handle relative paths
                if not os.path.isabs(folder_path):
                    folder_path = os.path.abspath(os.path.join(os.getcwd(), folder_path))
                
                # Check if the folder exists
                if not os.path.exists(folder_path):
                    print(f"\033[91mError: Folder not found: {folder_path}\033[0m")
                    continue
                
                if not os.path.isdir(folder_path):
                    print(f"\033[91mError: Path is not a directory: {folder_path}\033[0m")
                    continue
                
                # List PDF files in the folder
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                
                if not pdf_files:
                    print(f"\033[91mNo PDF files found in the folder: {folder_path}\033[0m")
                    continue
                
                print(f"\n\033[1m📚 Found {len(pdf_files)} PDF files in: {folder_path}\033[0m")
                
                # Create paper results similar to search results
                folder_results = []
                
                for i, pdf_file in enumerate(pdf_files):
                    file_path = os.path.join(folder_path, pdf_file)
                    # Extract title from filename (remove extension and replace underscores/hyphens)
                    title = os.path.splitext(pdf_file)[0].replace('_', ' ').replace('-', ' ')
                    
                    # Create a search result-like object for the file
                    paper = {
                        "title": title,
                        "file_path": file_path,
                        "url": f"file://{file_path}",
                        "source": "local_folder",
                        "summary": f"Local PDF file: {pdf_file}"
                    }
                    
                    folder_results.append(paper)
                    print(f"  {i+1}. {title}")
                
                # Store the results for discussion
                lab.last_search_results = folder_results
                lab.last_command_was_search = True
                
                # Add the command to the conversation
                search_message = f"I've loaded {len(folder_results)} papers from folder: {folder_path}"
                lab.user_message(search_message)
                
                # Display instructions for continuing with paper discussion
                print("\n\033[93mType 'continue' to have agents read and discuss these papers.\033[0m")
                continue
            
            # Check for search command
            if user_input.lower().startswith('/search'):
                # Extract query and options
                parts = user_input[7:].strip().split('--')
                query = parts[0].strip()
                
                if not query:
                    print("\033[93mPlease provide a search query: /search <query> [--type arxiv|biorxiv] [--results <number>] [--months <number>]\033[0m")
                    continue
                
                # Parse options
                search_type = "arxiv"  # default
                num_results = 3  # default
                months = 12  # default for biorxiv searches
                
                for part in parts[1:]:
                    if part.startswith('type '):
                        search_type = part[5:].strip()
                    elif part.startswith('results '):
                        try:
                            num_results = int(part[8:].strip())
                        except ValueError:
                            pass
                    elif part.startswith('months '):
                        try:
                            months = int(part[7:].strip())
                        except ValueError:
                            pass
                
                # Find an appropriate agent
                agent = lab.agents[0]  # default to first agent
                for a in lab.agents:
                    if search_type == "arxiv" and ("Biologist" in a.specialty or "Physicist" in a.specialty):
                        agent = a
                        break
                    elif search_type == "biorxiv" and "Biologist" in a.specialty:
                        agent = a
                        break
                
                # Perform search
                if search_type.lower() == "biorxiv":
                    print(f"\n\033[1m🔍 Searching {search_type} for papers published in the last {months} months matching: {query}\033[0m")
                    results = agent.search_web(query, search_type, num_results, months=months)
                else:
                    print(f"\n\033[1m🔍 Searching {search_type} for: {query}\033[0m")
                    results = agent.search_web(query, search_type, num_results)
                
                # Store the results for potential "continue" command
                lab.last_search_results = results
                lab.last_command_was_search = True
                
                # Display results
                print(f"\n\033[1m✅ Search Results ({len(results)} found)\033[0m")
                for i, result in enumerate(results):
                    print(f"\n\033[1;36m{i+1}. {result.get('title', 'Untitled')}\033[0m")
                    if 'url' in result:
                        print(f"   \033[94m{result['url']}\033[0m")
                    # Display publication date if available
                    if 'published' in result and result['published']:
                        print(f"   Published: {result['published']}")
                    elif result.get('source') == 'biorxiv' and 'url' in result:
                        # For bioRxiv, extract date from URL as fallback (YYYY.MM.DD)
                        url = result['url']
                        url_date_match = re.search(r'/10\.1101/(\d{4}\.\d{2}\.\d{2})\.', url)
                        if url_date_match:
                            date_str = url_date_match.group(1).replace('.', '-')
                            # Add version if present
                            version = ""
                            version_match = re.search(r'(v\d+)$', url)
                            if version_match:
                                version = f" ({version_match.group(1)})"
                            print(f"   Published: {date_str}{version}")
                    # Display snippet or summary
                    if 'snippet' in result and result['snippet'] and isinstance(result['snippet'], str):
                        print(f"   {result['snippet'][:200]}...")
                    elif 'summary' in result and result['summary'] and isinstance(result['summary'], str):
                        print(f"   {result['summary'][:200]}...")
                    elif 'content' in result and result['content'] and isinstance(result['content'], str):
                        print(f"   {result['content'][:200]}...")
                    elif 'content' in result and result['content']:
                        # If content exists but isn't a string, convert it
                        print(f"   {str(result['content'])[:200]}...")
                
                # Add the search to the conversation
                search_message = f"I've searched {search_type} for '{query}' and found {len(results)} results."
                lab.user_message(search_message)
                
                # Display instructions for continuing with paper discussion
                if len(results) > 0:
                    print("\n\033[93mType 'continue' to have agents download, read the full text, and discuss these papers in depth.\033[0m")
                
                continue
                
                
            # Check for cost command
            if user_input.lower() == '/cost':
                display_cost_estimate(lab)
                continue
            
            # Special handling for "continue" after a search
            if user_input.lower() == "continue" and lab.last_command_was_search and lab.last_search_results:
                # Have agents read and discuss the papers from search results
                print("\n\033[1m🧠 Starting in-depth discussion of search results...\033[0m")
                lab.discuss_search_results()
                
                # Reset search flag after handling
                lab.last_command_was_search = False
                continue
            else:
                # Reset search flag for other inputs
                lab.last_command_was_search = False
            
            # Add user message to conversation
            lab.user_message(user_input)
            
            
            # Generate and display responses from agents with enhanced collaboration
            # Increase the number of responses to encourage more agent interaction
            # (response display is handled within the method)
            num_responses = min(4, len(lab.agents))
            lab.generate_responses(num_responses, enable_collaboration=True)
            
            # Show prompt for next user input with clear formatting
            print("\n\033[1m> Your response:\033[0m")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except EOFError:
            print("\nInput terminated. Ending session.")
            break
        except Exception as e:
            import traceback
            print(f"\nAn error occurred: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            print("\nContinuing with a new prompt...")
            continue


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the application.
    
    Args:
        args: Command line arguments.
    """
    parser = argparse.ArgumentParser(description="AI Laboratory for scientific discussions")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args(args)
    run_lab(args.config)


if __name__ == "__main__":
    main()