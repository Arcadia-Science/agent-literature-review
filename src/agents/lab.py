"""
Laboratory module for managing a group of AI agents.
"""
import os
import random
from typing import Dict, List, Optional, Any, Tuple
import time
import threading
import sys
import itertools
import re
import textwrap
from .base import Agent
from ..config import LabConfig
from ..utils.document_loader import load_document


class ProgressIndicator:
    """A progress indicator that shows a spinner and message while agents are thinking."""
    
    def __init__(self, agent_name: str = None, message: str = None):
        """
        Initialize the progress indicator.
        
        Args:
            agent_name: The name of the agent that's thinking.
            message: The message to display.
        """
        self.agent_name = agent_name
        self.message = message or "Thinking"
        self.running = False
        self.spinner_thread = None
        self.spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
        
    def start(self):
        """Start the progress indicator."""
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        
    def stop(self):
        """Stop the progress indicator and clear the line."""
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * 100 + '\r')  # Clear the line with more spaces
        sys.stdout.flush()
        
    def _spin(self):
        """Display the spinner animation."""
        agent_prefix = f"{self.agent_name} " if self.agent_name else ""
        while self.running:
            for _ in range(10):  # Update approximately every 0.5 seconds
                if not self.running:
                    break
                time.sleep(0.05)
            if self.running:
                # Use a more subtle spinner display
                sys.stdout.write(f"\r\033[90m[{agent_prefix}{self.message}... {next(self.spinner)}]\033[0m")
                sys.stdout.flush()


class Laboratory:
    """A virtual laboratory of AI agents for scientific discussion."""
    
    def __init__(self, config: LabConfig):
        """
        Initialize the laboratory with a configuration.
        
        Args:
            config: Laboratory configuration.
        """
        self.config = config
        self.user_name = config.user_name
        self.agents = self._create_agents()
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Create a list of all known agents for reference
        self.known_agents = []
        for agent_config in config.agents:
            self.known_agents.append({
                "name": agent_config['name'],
                "specialty": agent_config['specialty']
            })
        
        # Set up agent collaborative network
        self._setup_agent_network()
        
        # Track active collaboration threads
        self.collaboration_threads = []
        
        # Track agents who were mentioned but didn't respond (priority queue for next turn)
        self.priority_response_queue = set()
        
        # Create a mention tracking system
        self.agent_mentions = {}  # Maps agent names to who mentioned them
        
        # Track if we're in a direct addressing flow (e.g., code execution fix)
        self.direct_addressing_flow = False
        self.direct_addressed_agent = None
        
        # Track last search results and whether the last command was a search
        self.last_search_results = []
        self.last_command_was_search = False
        
        # Paper cache to avoid downloading the same papers multiple times
        # Maps paper_id (arxiv_id or url) to content
        self.paper_cache = {}
    
    def _create_agents(self) -> List[Agent]:
        """
        Create agent instances based on configuration.
        
        Returns:
            List of initialized agents.
        """
        agents = []
        for agent_config in self.config.agents:
            agent = Agent(
                name=agent_config['name'],
                specialty=agent_config['specialty'],
                description=agent_config.get('description', f"Expert in {agent_config['specialty']}"),
                api_key=self.config.anthropic_api_key,
                model=self.config.model
            )
            agents.append(agent)
        return agents
        
    def get_cost_estimate(self) -> Dict[str, Any]:
        """
        Get the estimated cost of the laboratory session.
        
        Returns:
            Dictionary with cost estimate information.
        """
        return Agent.get_cost_estimate()
        
    def _setup_agent_network(self) -> None:
        """
        Connect agents to each other to enable direct communication.
        And provide each agent with knowledge of all other agents.
        """
        # Create a list of all agents with their details
        all_agents_info = []
        for agent in self.agents:
            all_agents_info.append({
                "name": agent.name,
                "specialty": agent.specialty
            })
            
        # Update each agent's known_agents list and connect them
        for agent in self.agents:
            # Set known agents
            agent.known_agents = all_agents_info
            # Update system message with new agent information
            agent.system_message = agent._create_system_message()
            
            # Connect to other agents
            for other_agent in self.agents:
                if agent != other_agent:
                    agent.connect_to_agent(other_agent)
    
    def add_message(self, content: str, sender: str, role: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            content: Message content.
            sender: Name of the sender.
            role: Role of the sender (user or assistant).
        """
        message = {
            "role": role,
            "content": content,
            "name": sender
        }
        self.conversation_history.append(message)
        
        # Also add to each agent's conversation history
        for agent in self.agents:
            agent.add_to_conversation(role, content, sender)
    
    def user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            content: Message content from the user.
        """
        self.add_message(content, self.user_name, "user")
        
        # ANSI color codes
        USER_COLOR = "\033[97m"  # White for user
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        # Format and print user message
        lines = content.split('\n')
        width = 110  # Increased width of the box
        
        # Print header with user's actual name
        print(f"{USER_COLOR}┌─── {BOLD}{self.user_name}{RESET}{USER_COLOR} {'─' * (width - len(self.user_name) - 6)}┐{RESET}")
        
        # Check for agent mentions and mark them
        direct_mentions = {}
        # Create a mapping of agents to consistent colors (duplicate from generate_responses)
        COLORS = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        
        agent_colors = {}
        available_colors = list(COLORS.keys())[:-2]  # Exclude reset and bold
        for idx, agent in enumerate(self.agents):
            agent_colors[agent.name] = available_colors[idx % len(available_colors)]
        
        # Detect mentions of agents with improved matching
        for agent in self.agents:
            # First, check standard format with colon: @AgentName:
            if f"@{agent.name}:".lower() in content.lower():
                direct_mentions[agent.name] = COLORS[agent_colors[agent.name]]
                
            # Also check mentions without colon: @AgentName
            elif f"@{agent.name}".lower() in content.lower():
                # Verify it's a proper mention
                mention_pattern = f"@{agent.name}"
                index = content.lower().find(mention_pattern.lower())
                pattern_len = len(mention_pattern)
                is_valid_start = index == 0 or content[index-1] in " \n\t"
                is_valid_end = (index + pattern_len >= len(content) or 
                              content[index + pattern_len] in " \n\t.,;:!?")
                
                if is_valid_start and is_valid_end:
                    direct_mentions[agent.name] = COLORS[agent_colors[agent.name]]
        
        # Print message content with word wrap and agent highlights
        for line in lines:
            while line:
                if len(line) <= width - 4:  # -4 for margins
                    # Check for direct mentions and highlight them
                    highlighted_line = line
                    for mentioned_agent, mention_color in direct_mentions.items():
                        # Check for different forms of mentions and highlight them
                        # First, standard format with colon
                        mention_pattern = f"@{mentioned_agent}:"
                        if mention_pattern.lower() in highlighted_line.lower():
                            # Find all occurrences with case-insensitive search
                            pattern_len = len(mention_pattern)
                            pos = 0
                            while True:
                                pos = highlighted_line.lower().find(mention_pattern.lower(), pos)
                                if pos == -1:
                                    break
                                # Extract actual text as it appears in the original
                                actual_text = highlighted_line[pos:pos+pattern_len]
                                # Replace with highlighted version
                                highlighted_line = highlighted_line.replace(
                                    actual_text, 
                                    f"{RESET}{mention_color}{BOLD}@{mentioned_agent}{RESET}{mention_color}:{RESET}{USER_COLOR}"
                                )
                                pos += pattern_len
                        
                        # Also check for mentions without colon
                        mention_pattern = f"@{mentioned_agent}"
                        if mention_pattern.lower() in highlighted_line.lower():
                            # Check if it's a proper mention (not already handled with colon format)
                            # and not part of another word
                            pattern_len = len(mention_pattern)
                            pos = 0
                            while True:
                                pos = highlighted_line.lower().find(mention_pattern.lower(), pos)
                                if pos == -1:
                                    break
                                # Extract actual text as it appears in the original
                                actual_text = highlighted_line[pos:pos+pattern_len]
                                # Verify it's a proper mention with appropriate boundaries
                                is_valid_start = pos == 0 or highlighted_line[pos-1] in " \n\t"
                                next_pos = pos + pattern_len
                                is_valid_end = (next_pos >= len(highlighted_line) or 
                                               highlighted_line[next_pos] in " \n\t.,;:!?")
                                
                                if is_valid_start and is_valid_end:
                                    # Replace with highlighted version
                                    highlighted_line = highlighted_line.replace(
                                        actual_text, 
                                        f"{RESET}{mention_color}{BOLD}{actual_text}{RESET}{USER_COLOR}"
                                    )
                                pos += 1
                    
                    print(f"{USER_COLOR}│ {highlighted_line}{RESET}")
                    line = ""
                else:
                    # Find a good breaking point
                    break_point = width - 4
                    while break_point > 0 and line[break_point] != ' ':
                        break_point -= 1
                    if break_point == 0:  # No space found, hard break
                        break_point = width - 4
                    
                    # Get the current line segment and highlight any mentions
                    line_segment = line[:break_point]
                    for mentioned_agent, mention_color in direct_mentions.items():
                        mention_pattern = f"@{mentioned_agent}:"
                        if mention_pattern in line_segment:
                            # Highlight the mention
                            line_segment = line_segment.replace(
                                mention_pattern, 
                                f"{RESET}{mention_color}{BOLD}@{mentioned_agent}{RESET}{mention_color}:{RESET}{USER_COLOR}"
                            )
                    
                    print(f"{USER_COLOR}│ {line_segment}{RESET}")
                    line = line[break_point:].lstrip()
        
        # Add indicators for directly addressed agents in a more subtle way
        direct_message_indicators = []
        for agent_name in direct_mentions:
            mention_color = direct_mentions[agent_name]
            direct_message_indicators.append(f"{mention_color}{agent_name}{RESET}")
        
        # Print footer with indicators
        if direct_message_indicators:
            if len(direct_message_indicators) == 1:
                direct_to = f"To: {direct_message_indicators[0]}"
            else:
                direct_to = "To: " + ", ".join(direct_message_indicators)
            print(f"{USER_COLOR}└{'─' * (width - len(direct_to) - 3)}[ {direct_to} {USER_COLOR}]─┘{RESET}")
        else:
            print(f"{USER_COLOR}└{'─' * width}┘{RESET}")
            
        print()  # Add spacing between messages
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """
        Get an agent by name.
        
        Args:
            name: Name of the agent to find.
            
        Returns:
            The agent if found, None otherwise.
        """
        for agent in self.agents:
            if agent.name.lower() == name.lower():
                return agent
        return None
    
    def process_documents(self, query: str) -> Dict[str, str]:
        """
        Process documents in the documents directory.
        
        Args:
            query: The query to guide document processing.
            
        Returns:
            A dictionary mapping document names to agent summaries.
        """
        if not self.config.documents_dir or not os.path.exists(self.config.documents_dir):
            return {"error": f"Documents directory not found: {self.config.documents_dir}"}
        
        results = {}
        documents = [f for f in os.listdir(self.config.documents_dir) if f.endswith(('.pdf', '.txt', '.md'))]
        
        if not documents:
            return {"error": f"No supported documents found in {self.config.documents_dir}"}
        
        # Distribute documents among agents
        docs_per_agent = max(1, len(documents) // len(self.agents))
        random.shuffle(documents)
        
        agent_docs = {}
        for i, agent in enumerate(self.agents):
            start_idx = i * docs_per_agent
            end_idx = min(start_idx + docs_per_agent, len(documents))
            agent_docs[agent.name] = documents[start_idx:end_idx]
        
        # Process each document with its assigned agent
        for agent_name, doc_list in agent_docs.items():
            agent = self.get_agent_by_name(agent_name)
            if not agent:
                continue
                
            for doc in doc_list:
                doc_path = os.path.join(self.config.documents_dir, doc)
                content = load_document(doc_path)
                if content:
                    # Show progress indicator
                    progress = ProgressIndicator(agent_name, f"analyzing document: {doc}")
                    progress.start()
                    
                    try:
                        print(f"\n{agent_name} is analyzing '{doc}'...")
                        
                        # Create metadata about the document
                        document_metadata = {
                            "title": os.path.basename(doc_path),
                            "source": "Local Document"
                        }
                        
                        # Try to extract more metadata from the filename or path
                        filename = os.path.basename(doc_path)
                        # Extract potential authors from filename (assuming Author_Title format)
                        if "_" in filename:
                            parts = filename.split("_", 1)
                            if len(parts) > 0:
                                # Only use before underscore as author if it looks like a name
                                potential_author = parts[0].replace("-", " ")
                                if re.match(r"^[A-Za-z\s]+$", potential_author) and len(potential_author) > 3:
                                    document_metadata["authors"] = potential_author
                        
                        # Extract date if present in filename (looking for YYYY or YYYYMMDD formats)
                        date_match = re.search(r"(19|20)\d{2}(\d{4})?", filename)
                        if date_match:
                            document_metadata["date"] = date_match.group(0)
                        
                        summary = agent.process_document(content, query, document_metadata)
                        results[doc] = {
                            "agent": agent_name,
                            "summary": summary
                        }
                    finally:
                        progress.stop()
        
        return results
    
    def get_directly_addressed_agents(self, message: Optional[str] = None) -> List[Agent]:
        """
        Determine which agents were directly addressed in a message using @AgentName: syntax.
        If no message is provided, checks the last message in the conversation history.
        
        Args:
            message: The message to check for direct mentions. If None, uses last message.
            
        Returns:
            List of Agent objects that were directly addressed.
        """
        if not message and self.conversation_history:
            # Use the last message in the conversation
            message = self.conversation_history[-1].get("content", "")
            
        addressed_agents = []
        
        if message:
            # First, try with standard format: @AgentName:
            for agent in self.agents:
                mention_pattern = f"@{agent.name}:"
                if mention_pattern.lower() in message.lower():
                    # Verify this is an actual direct mention, not part of another word
                    index = message.lower().find(mention_pattern.lower())
                    # Check if it's at the start of the message or preceded by whitespace/newline
                    if index == 0 or message[index-1] in " \n\t":
                        addressed_agents.append(agent)
                        print(f"\033[1;33m🔍 Found direct mention of {agent.name} with @AgentName: syntax\033[0m")
            
            # If no agents found with the standard format, try with a looser format: @AgentName
            if not addressed_agents:
                for agent in self.agents:
                    mention_pattern = f"@{agent.name}"
                    # Don't include the colon so we can match formats like @AgentName or @AgentName,
                    if mention_pattern.lower() in message.lower():
                        # Verify this is an actual direct mention, not part of another word
                        index = message.lower().find(mention_pattern.lower())
                        # Check if it's at the start of the message or preceded by whitespace/newline
                        # and followed by space, punctuation, or end of message
                        pattern_len = len(mention_pattern)
                        is_valid_start = index == 0 or message[index-1] in " \n\t"
                        is_valid_end = (index + pattern_len >= len(message) or 
                                       message[index + pattern_len] in " \n\t.,;:!?")
                        
                        if is_valid_start and is_valid_end:
                            addressed_agents.append(agent)
                            print(f"\033[1;33m🔍 Found direct mention of {agent.name} with @AgentName syntax\033[0m")
        
        return addressed_agents

    def generate_responses(self, num_responses: int = 2, research_query: Optional[str] = None, 
                          enable_collaboration: bool = True, show_conversation_map: bool = True) -> List[Tuple[str, str]]:
        """
        Generate responses from random agents.
        
        Args:
            num_responses: Number of agent responses to generate.
            research_query: Optional query to research before responding.
            enable_collaboration: Whether to enable agent cross-talk and collaboration.
            show_conversation_map: Whether to show a visual conversation map at the end.
            
        Returns:
            List of (agent_name, response) tuples.
        """
        # ANSI color codes for better visualization
        COLORS = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        
        responses = []
        
        # Check if any agents were directly addressed in the last message
        directly_addressed = self.get_directly_addressed_agents()
        
        # Check if the last message has direct addressing (@AgentName syntax)
        is_direct_addressing = False
        direct_addressing_agents = []
        if self.conversation_history:
            last_message = self.conversation_history[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                
                # First try standard format with colon: @AgentName:
                for agent in self.agents:
                    mention_pattern = f"@{agent.name}:"
                    if mention_pattern.lower() in content.lower():
                        index = content.lower().find(mention_pattern.lower())
                        if index == 0 or (index > 0 and content[index-1] in " \n\t"):
                            is_direct_addressing = True
                            direct_addressing_agents.append(agent.name)
                
                # If no mentions found with colon, try looser format: @AgentName
                if not direct_addressing_agents:
                    for agent in self.agents:
                        mention_pattern = f"@{agent.name}"
                        if mention_pattern.lower() in content.lower():
                            index = content.lower().find(mention_pattern.lower())
                            pattern_len = len(mention_pattern)
                            is_valid_start = index == 0 or content[index-1] in " \n\t"
                            is_valid_end = (index + pattern_len >= len(content) or 
                                          content[index + pattern_len] in " \n\t.,;:!?")
                            
                            if is_valid_start and is_valid_end:
                                is_direct_addressing = True
                                direct_addressing_agents.append(agent.name)
                
                # Debug output to confirm direct addressing detection
                if is_direct_addressing:
                    print(f"\n\033[1;33m🔍 Direct addressing detected in user message to: {', '.join(direct_addressing_agents)}\033[0m")
        
        # First check if there are priority agents (mentioned but didn't respond previously)
        priority_agents = []
        if self.priority_response_queue:
            print(f"\nPrioritizing agents who were mentioned but didn't respond: {', '.join(self.priority_response_queue)}")
            for agent_name in self.priority_response_queue:
                agent = self.get_agent_by_name(agent_name)
                if agent:
                    priority_agents.append(agent)
            
            # Clear the priority queue after handling
            self.priority_response_queue.clear()
            
        # If agents were directly addressed, they should respond
        if directly_addressed:
            # Combine priority agents with directly addressed agents
            responding_agents = list(set(priority_agents + directly_addressed))
            # Limit to the number of responses requested
            responding_agents = responding_agents[:min(num_responses, len(responding_agents))]
            print(f"\nDirectly addressed agents will respond: {', '.join([a.name for a in responding_agents])}")
        elif priority_agents:
            # If we have priority agents but no direct addressing, use them plus some random ones
            remaining_spots = max(0, num_responses - len(priority_agents))
            if remaining_spots > 0:
                # Add random agents that aren't already in priority_agents
                available_agents = [a for a in self.agents if a not in priority_agents]
                if available_agents:
                    random_agents = random.sample(available_agents, min(remaining_spots, len(available_agents)))
                    responding_agents = priority_agents + random_agents
                else:
                    responding_agents = priority_agents
            else:
                responding_agents = priority_agents[:num_responses]
        elif is_direct_addressing:
            # If user is using @AgentName: format but no agents were detected, 
            # this likely means addressing a specific agent didn't work.
            # Only let directly addressed agents respond.
            responding_agents = directly_addressed
            
            # Log the direct addressing agents we detected
            print(f"\n\033[1;33m🔍 Direct address check: User mentioned {', '.join(direct_addressing_agents)}\033[0m")
            # Log which agents are in the responding list
            if responding_agents:
                print(f"\n\033[1;33m🔍 Responding agents: {', '.join([a.name for a in responding_agents])}\033[0m")
            else:
                print(f"\n\033[1;33m🔍 No responding agents matched in initial check\033[0m")
            
            if not responding_agents:
                # Try to find the agents directly in the message text as a fallback
                last_message_content = self.conversation_history[-1].get("content", "")
                for agent in self.agents:
                    mention_pattern = f"@{agent.name}"
                    if mention_pattern.lower() in last_message_content.lower():
                        # This is a direct mention but perhaps not in the exact format expected
                        print(f"\n\033[1;33m🔍 Found direct mention of {agent.name} in alternative format\033[0m")
                        responding_agents.append(agent)
                
                if not responding_agents:
                    print("\nUser addressed specific agents, but no valid agents were found. No responses will be generated.")
                    return []
        else:
            # Otherwise, randomly select agents to respond, without repetition if possible
            responding_agents = random.sample(self.agents, min(num_responses, len(self.agents)))
        
        # Create a mapping of agents to consistent colors
        agent_colors = {}
        available_colors = list(COLORS.keys())[:-2]  # Exclude reset and bold
        for idx, agent in enumerate(self.agents):
            agent_colors[agent.name] = available_colors[idx % len(available_colors)]
        
        # Track which agents have participated
        participating_agents = set()
        
        # First generate primary responses
        primary_responses = []
        for agent in responding_agents:
            # Start progress indicator for this agent
            progress = ProgressIndicator(agent.name, "thinking")
            progress.start()
            
            try:
                if research_query:
                    response = agent.generate_response_with_research(research_query)
                else:
                    response = agent.generate_response()
                
                self.add_message(response, agent.name, "assistant")
                primary_responses.append((agent, response))
                participating_agents.add(agent.name)
            finally:
                # Always stop the progress indicator, even if an error occurs
                progress.stop()
        
        # Track who was mentioned by whom with enhanced tracking
        mentioned_agents = set()
        agent_mentions = {}
        
        # Identify valid direct mentions in primary responses with improved extraction
        for agent, response in primary_responses:
            agent_name = agent.name
            
            # Extract all agents mentioned in this response
            mentioned = self._extract_mentioned_agents(response)
            agent_mentions[agent_name] = mentioned
            mentioned_agents.update(mentioned)
            
            # Track mentions in our conversation history for future turns
            for mentioned_name in mentioned:
                if mentioned_name not in self.agent_mentions:
                    self.agent_mentions[mentioned_name] = []
                if agent_name not in self.agent_mentions[mentioned_name]:
                    self.agent_mentions[mentioned_name].append(agent_name)
        
        # Process responses for direct messages if collaboration is enabled
        if enable_collaboration:
            # New adaptive collaboration approach:
            # 1. Always allow mentioned agents to respond
            # 2. Encourage chained dialogue between agents
            # 3. Structure collaboration like natural laboratory conversations
            
            # First priority: get responses from explicitly mentioned agents
            # ALWAYS include all mentioned agents - they should ALWAYS respond when directly addressed
            mentioned_to_respond = set()
            
            # IMPORTANT: Check if we're in a code execution/fix flow or direct addressing
            # First check if we're in a direct addressing flow (e.g., code execution fix)
            if self.direct_addressing_flow and self.direct_addressed_agent:
                # Only the specified agent should respond
                print(f"\n\033[1;33m🔍 Direct addressing flow active - only allowing responses from {self.direct_addressed_agent}\033[0m")
                mentioned_to_respond = {self.direct_addressed_agent}
                
                # Reset the flow after using it once, so it doesn't persist indefinitely
                self.direct_addressing_flow = False
            # Otherwise check if this was a direct address in the user message
            elif is_direct_addressing and len(direct_addressing_agents) > 0:
                # Only the directly addressed agents should respond
                print(f"\n\033[1;33m🔍 Direct addressing detected - only allowing responses from {', '.join(direct_addressing_agents)}\033[0m")
                mentioned_to_respond = set(direct_addressing_agents)
            else:
                # Normal conversation flow - process all mentions
                for sender, recipients in agent_mentions.items():
                    for recipient in recipients:
                        # Always add mentioned agents, even if they already participated
                        # This ensures agents don't ignore direct questions
                        mentioned_to_respond.add(recipient)
            
            # Second priority (NEW): have some agents respond to the most active agent
            # This creates more natural back-and-forth conversations
            # But ONLY if this wasn't a direct address to specific agents
            if len(primary_responses) > 0 and not is_direct_addressing:
                # Find the agent with the most substantive response (rough heuristic)
                most_active_agent = max(primary_responses, key=lambda x: len(x[1]))
                most_active_name = most_active_agent[0].name
                
                # Find agents who might want to respond to this active agent
                potential_responders = []
                for agent in self.agents:
                    # Only consider agents who haven't participated yet
                    if agent.name not in participating_agents and agent.name != most_active_name:
                        potential_responders.append(agent.name)
                
                # Randomly select a few potential responders
                if potential_responders:
                    num_additional = min(2, len(potential_responders))
                    additional_responders = random.sample(potential_responders, num_additional)
                    
                    # For each selected responder, create a "synthetic mention"
                    for responder_name in additional_responders:
                        # Add them to the list of agents to respond
                        mentioned_to_respond.add(responder_name)
                        
                        # Add a synthetic mention so we know who they should respond to
                        if most_active_name not in agent_mentions:
                            agent_mentions[most_active_name] = set()
                        agent_mentions[most_active_name].add(responder_name)
            
            # Increase the number of follow-up responses to encourage rich collaboration
            # But only if this isn't a direct address
            max_followups = min(len(mentioned_to_respond), 5)  # Allow up to 5 follow-up responses
            
            # If we are in direct addressing mode (user direct messaging or code execution flow),
            # strictly enforce only those agents respond
            if self.direct_addressing_flow and self.direct_addressed_agent:
                # Only allow the agent in our direct addressing flow
                mentioned_to_respond = {name for name in mentioned_to_respond 
                                     if name == self.direct_addressed_agent}
                print(f"\n\033[1;33m🔍 Direct addressing flow: Only allowing responses from {self.direct_addressed_agent}\033[0m")
            elif is_direct_addressing:
                # Filter mentioned_to_respond to only include directly addressed agents
                mentioned_to_respond = {name for name in mentioned_to_respond 
                                     if name in direct_addressing_agents}
                print(f"\n\033[1;33m🔍 Direct addressing: Only allowing responses from {', '.join(mentioned_to_respond)}\033[0m")
                
                # Handle direct addressing to PI for report requests
                for mentioned_name in mentioned_to_respond:
                    mentioned_agent = self.get_agent_by_name(mentioned_name)
                    if mentioned_agent and "PI" in mentioned_agent.specialty:
                        # Get the last user message to check for report request
                        if self.conversation_history:
                            last_message = self.conversation_history[-1]
                            if last_message.get("role") == "user":
                                user_content = last_message.get("content", "")
                                # Check if this is a report request (contains any of the report request phrases)
                                report_phrases = ["write report", "write the report", "generate report", "create report", 
                                               "save report", "write that report", "save that report", "write to disk"]
                                if any(phrase in user_content.lower() for phrase in report_phrases):
                                    # Extract topic from user message - remove the agent mention and report phrase
                                    topic = user_content
                                    for phrase in report_phrases:
                                        if phrase in topic.lower():
                                            topic = topic.lower().replace(phrase, "").strip()
                                    # Remove the agent mention
                                    mention_pattern = f"@{mentioned_name}"
                                    topic = topic.replace(mention_pattern, "").replace(":", "").strip()
                                    
                                    # If topic is empty, use a generic title
                                    if not topic:
                                        topic = "Research Discussion Summary"
                                    
                                    print(f"\n\033[1;33m🔍 Report request detected for topic: {topic}\033[0m")
                                    # Pass a context that explicitly flags this as a report request for the PI agent
                                    context = f"""This is a direct request for you to write a report to disk about: {topic}
                                    
The user has specifically asked you to save this report to disk."""
            
            if max_followups > 0:
                # Determine which agents will respond
                if len(mentioned_to_respond) > max_followups:
                    mentioned_to_respond = set(random.sample(list(mentioned_to_respond), max_followups))
                
                # Generate responses for these mentioned agents
                for mentioned_name in mentioned_to_respond:
                    mentioned_agent = self.get_agent_by_name(mentioned_name)
                    if not mentioned_agent:
                        continue
                    
                    # Determine who mentioned this agent (or who they should respond to)
                    mentioners = [sender for sender, recipients in agent_mentions.items() if mentioned_name in recipients]
                    
                    # If no valid mentioners (shouldn't happen), skip
                    if not mentioners:
                        continue
                    
                    # Pick the first mentioner (or random one if multiple)
                    mentioner = random.choice(mentioners)
                    
                    # Find the actual message from the mentioner in the conversation history
                    mentioner_message = ""
                    for message in reversed(self.conversation_history):
                        if message.get("role") == "assistant" and message.get("name") == mentioner:
                            mentioner_message = message.get("content", "")
                            break
                    
                    # Extract just the part that might be addressed to this agent
                    direct_mention = f"@{mentioned_name}:"
                    mention_content = ""
                    if direct_mention in mentioner_message:
                        parts = mentioner_message.split(direct_mention)
                        if len(parts) > 1:
                            # Get the content after the mention
                            after_mention = parts[1]
                            # Extract until the next @ mention or end of message
                            end_idx = len(after_mention)
                            for agent in self.known_agents:
                                if agent["name"] != mentioned_name:
                                    next_mention = f"@{agent['name']}:"
                                    next_idx = after_mention.find(next_mention)
                                    if next_idx != -1 and next_idx < end_idx:
                                        end_idx = next_idx
                            mention_content = after_mention[:end_idx].strip()
                    
                    # Use the full message if no direct mention found or can't extract properly
                    specific_content = mention_content if mention_content else mentioner_message
                    
                    # Check if this agent is in our priority queue (was mentioned before but didn't respond)
                    is_priority_response = mentioned_name in self.priority_response_queue
                    
                    # Create a more detailed context with conversation history for context
                    previous_content = ""
                    relevant_exchanges = []
                    
                    # Extract relevant conversation history between these agents
                    mentioner_messages = []
                    for message in reversed(self.conversation_history[-15:]):  # Look at recent messages
                        if message.get("role") == "assistant" and message.get("name") == mentioner:
                            mentioner_messages.append(message.get("content", ""))
                    
                    # Get 1-2 previous messages if available
                    if len(mentioner_messages) > 1:
                        previous_content = "\n\nTheir previous message: " + mentioner_messages[1][:200] + "..."
                    
                    # Extract technical concepts from recent conversation for topic continuity
                    technical_concepts = self._extract_technical_concepts(self.conversation_history[-15:])
                    
                    # Identify messages related to the current technical concepts to provide topic context
                    related_messages = []
                    for message in reversed(self.conversation_history[-20:]):
                        # Skip very short messages and focus on substantive ones
                        if len(message.get("content", "")) < 50:
                            continue
                            
                        # Check if this message contains any of our technical concepts
                        msg_content = message.get("content", "")
                        if any(concept in msg_content.lower() for concept in technical_concepts):
                            # This message is topic-relevant
                            sender = message.get("name", "Unknown")
                            # Create a short excerpt with the relevant concept highlighted
                            excerpt = f"{sender}: {self._create_excerpt(msg_content, technical_concepts)}"
                            related_messages.append(excerpt)
                    
                    # Format into coherent topic background (limit to avoid overwhelming)
                    topic_context = "\n\n".join(related_messages[:5])
                    
                    # Generate structured context for laboratory-style dialogue with rich reference context
                    context = f"""You are participating in a collaborative laboratory discussion about {', '.join(technical_concepts) if technical_concepts else 'scientific research'}. 

Your colleague {mentioner} specifically mentioned you in the following message:

---
{specific_content}
---
{previous_content}

IMPORTANT TOPIC BACKGROUND (recent discussion threads):
{topic_context}

This directly relates to your expertise as a {mentioned_agent.specialty}. You should respond thoroughly to their specific points and questions.

LABORATORY DIALOGUE INSTRUCTIONS:
1. Start your response with "@{mentioner}:" to directly engage with them
2. Reference specific details they mentioned (quote briefly if needed)
3. Either build on their point with additional insights, ask clarifying questions about a specific detail, or respectfully challenge an assumption with your own perspective
4. Be detailed and substantive - share specific examples, methods, or concepts from your {mentioned_agent.specialty} background
5. When appropriate, bring another colleague into the discussion using "@TheirName:" with a specific question
6. Ensure your response demonstrates understanding of the specific technical concepts being discussed

IMPORTANT: Your response should be thorough (4-6 sentences) and demonstrate your specific expertise. Create a natural back-and-forth dialogue that would occur in a real scientific meeting.
"""
                    
                    # Show progress indicator
                    progress = ProgressIndicator(mentioned_agent.name, f"formulating a response to {mentioner}")
                    progress.start()
                    
                    try:
                        # Check if we have a special report context for PI agents
                        if "PI" in mentioned_agent.specialty and "report request detected" in str(context).lower() and "This is a direct request for you to write a report to disk" in str(context):
                            # Extract the topic from the context
                            topic_match = re.search(r'write a report to disk about: (.+?)(?=\n|$)', context)
                            report_topic = topic_match.group(1).strip() if topic_match else "Research Discussion Summary"
                            
                            # Generate the scientific report
                            print(f"\n\033[1;33m🔍 Generating scientific report on: {report_topic}\033[0m")
                            report_data = mentioned_agent.generate_scientific_report(report_topic, self.conversation_history)
                            
                            # Save the report to disk
                            save_result = mentioned_agent.write_report_to_disk(report_data["title"], report_data["content"])
                            
                            # Generate a response about writing the report
                            direct_response = f"I've compiled our research discussion into a formal scientific report on '{report_data['title']}'. {save_result} The report includes our key findings, methodologies, and conclusions."
                        else:
                            # Generate a regular focused response
                            direct_response = mentioned_agent.generate_response_to_direct_message(
                                mentioner, 
                                context
                            )
                        
                        # Add the direct response to the conversation
                        self.add_message(direct_response, mentioned_agent.name, "assistant")
                        primary_responses.append((mentioned_agent, direct_response))
                        participating_agents.add(mentioned_agent.name)
                    finally:
                        # Always stop the progress indicator, even if an error occurs
                        progress.stop()
        
        # Format and display all responses
        for agent, response in primary_responses:
            # Skip messages that indicate they were for someone else
            if response.startswith("[This message was for"):
                continue
                
            responses.append((agent.name, response))
            
            # Print formatted response
            color = COLORS[agent_colors[agent.name]]
            reset = COLORS["reset"]
            bold = COLORS["bold"]
            
            # Format multiline messages
            lines = response.split('\n')
            width = 110  # Width of the box (increased for better readability)
            
            # Print header with agent name
            print(f"{color}┌─── {bold}{agent.name}{reset}{color} {'─' * (width - len(agent.name) - 6)}┐{reset}")
            
            # Print message content with word wrap
            for line in lines:
                while line:
                    if len(line) <= width - 4:  # -4 for margins
                        print(f"{color}│ {line}{reset}")
                        line = ""
                    else:
                        # Find a good breaking point
                        break_point = width - 4
                        while break_point > 0 and line[break_point] != ' ':
                            break_point -= 1
                        if break_point == 0:  # No space found, hard break
                            break_point = width - 4
                        
                        print(f"{color}│ {line[:break_point]}{reset}")
                        line = line[break_point:].lstrip()
            
            # Print footer
            print(f"{color}└{'─' * width}┘{reset}")
            print()  # Add spacing between messages
            
            # Add a short delay between agent responses to make the conversation feel more natural
            time.sleep(0.5)
        
        # Display conversation map if enabled
        if show_conversation_map and len(primary_responses) > 1:
            self._display_conversation_map(primary_responses, agent_colors, COLORS)
        
        return responses
        
    def _display_conversation_map(self, responses: List[Tuple[Any, str]], agent_colors: Dict[str, str], colors: Dict[str, str]) -> None:
        """
        Display a visual map of the conversation showing agent interactions.
        
        Args:
            responses: List of (agent, response) tuples.
            agent_colors: Mapping of agent names to color keys.
            colors: Dictionary of ANSI color codes.
        """
        reset = colors["reset"]
        bold = colors["bold"]
        
        # Track which agents are talking to which other agents
        conversation_graph = {}
        for agent, response in responses:
            sender = agent.name
            if sender not in conversation_graph:
                conversation_graph[sender] = set()
                
            # Find all mentioned agents in this response
            for other_agent in self.agents:
                if other_agent.name != sender and f"@{other_agent.name}:" in response:
                    conversation_graph[sender].add(other_agent.name)
        
        # Print conversation map header
        print("\n" + bold + "📊 CONVERSATION MAP" + reset)
        print("─" * 110)
        
        # Display each agent and who they spoke to
        for agent_name in conversation_graph:
            agent_color = colors[agent_colors[agent_name]]
            print(f"{agent_color}{bold}{agent_name}{reset} spoke to:")
            
            if conversation_graph[agent_name]:
                for recipient in conversation_graph[agent_name]:
                    recipient_color = colors[agent_colors[recipient]]
                    print(f"  {agent_color}→ {recipient_color}{recipient}{reset}")
            else:
                print(f"  {agent_color}(no direct mentions){reset}")
        
        # Display agents who were mentioned but didn't speak
        # Creating a more comprehensive tracking system
        mentioned_agents = set()
        responded_agents = set(conversation_graph.keys())
        
        # Find all agents who were mentioned by anyone
        for sender in conversation_graph:
            for recipient in conversation_graph[sender]:
                mentioned_agents.add(recipient)
        
        # Find agents who were mentioned but didn't respond
        non_responsive_agents = mentioned_agents - responded_agents
        
        # Improved display with counts and explanation
        if non_responsive_agents:
            print("\n" + bold + f"Agents who were addressed ({len(non_responsive_agents)}):" + reset)
            for agent_name in non_responsive_agents:
                # Find who mentioned this agent
                mentioners = []
                for sender, recipients in conversation_graph.items():
                    if agent_name in recipients:
                        mentioners.append(sender)
                
                agent_color = colors[agent_colors[agent_name]]
                mentioners_str = ", ".join(mentioners)
                print(f"  {agent_color}{agent_name}{reset} (mentioned by: {mentioners_str})")
                
                # Get this agent to ensure they respond next time
                mentioned_agent = self.get_agent_by_name(agent_name)
                if mentioned_agent:
                    # Add to priority queue for next turn
                    self.priority_response_queue.add(agent_name)
                    # Also track who mentioned them for context
                    if agent_name not in self.agent_mentions:
                        self.agent_mentions[agent_name] = []
                    for mentioner in mentioners:
                        self.agent_mentions[agent_name].append(mentioner)
                    print(f"    → Will prioritize {agent_name}'s response in next turn")
                
        print("─" * 110)
        
    def facilitate_agent_discussion(self, topic: str, rounds: int = 3, show_conversation_map: bool = True) -> List[Tuple[str, str]]:
        """
        Facilitate a focused discussion between agents on a specific topic.
        
        Args:
            topic: The topic for agents to discuss.
            rounds: Number of discussion rounds (each agent speaks once per round).
            show_conversation_map: Whether to show a visual conversation map at the end.
            
        Returns:
            List of (agent_name, response) tuples for all messages in the discussion.
        """
        print(f"\n\033[1m🔬 Starting agent discussion on: {topic}\033[0m\n")
        
        # Set up ANSI colors
        COLORS = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        
        # Create a mapping of agents to consistent colors
        agent_colors = {}
        available_colors = list(COLORS.keys())[:-2]  # Exclude reset and bold
        for idx, agent in enumerate(self.agents):
            agent_colors[agent.name] = available_colors[idx % len(available_colors)]
        
        
        # Standard team selection for all topics
        starter_team_size = min(3, len(self.agents))
        starter_team = random.sample(self.agents, starter_team_size)
        starter_agent = starter_team[0]
        
        # Create a structure for who should participate in each round
        participation_plan = {
            0: [starter_agent],  # First round just the starter agent
            1: [a for a in starter_team if a != starter_agent],  # Second round the rest of the starter team
        }
        
        # Track which agents have already participated to prevent repeats
        participating_agents = set([agent.name for agent in starter_team])
        
        # Build a focused context that emphasizes concrete outcomes and clear participation
        agents_str = ", ".join([f"{agent.name} ({agent.specialty})" for agent in starter_team])

        context = f"""I need help with this task: {topic}

This is a structured, focused discussion. I've selected these specialists to collaborate: {agents_str}

@{starter_agent.name}: Please provide a concise assessment of how you can contribute to this task. Focus on practical next steps drawing on your experience with {starter_agent.specialty}. Only mention other specialists if their expertise is essential for critical aspects.
"""
        
        # Add the discussion context as a user message
        self.add_message(context, self.user_name, "user")
        
        # Show progress indicator
        progress = ProgressIndicator(starter_agent.name, "researching and formulating an initial response")
        progress.start()
        
        try:
            # Create a prompt for the first agent
            starter_context = f"""You are {starter_agent.name} contributing to a casual lab discussion.

CRITICAL: NEVER introduce yourself as "{starter_agent.name}, a {starter_agent.specialty}" or any variation.
NEVER begin with "As a researcher in..." or "From my perspective as..."
Simply dive straight into your substantive thoughts about the problem at hand.

Start with a natural opening like:
- "I think we should approach this by..."
- "The key challenge here is..."
- "Based on my previous work, we could..."

Keep your tone natural, conversational, and first-person."""

            # Generate initial response with specialized guidance
            specialized_messages = [{"role": "system", "content": starter_context}]
            initial_response = starter_agent.generate_response_with_research(topic)
            
            
            # Add the final response to the conversation
            self.add_message(initial_response, starter_agent.name, "assistant")
        finally:
            progress.stop()
        
        # Format and print the response
        self._print_agent_message(starter_agent.name, initial_response, agent_colors, COLORS)
        
        all_responses = [(starter_agent.name, initial_response)]
        
        # Track which agents were explicitly mentioned to avoid endless loops
        mentioned_agents = self._extract_mentioned_agents(initial_response)
        
        
        # Conduct discussion rounds
        for round_num in range(rounds):
            # Add task-focused transitions between rounds
            if round_num == 0:
                # For round 1, explicitly direct specific agents to respond
                # Get names of next agents to participate from participation plan
                next_agents = participation_plan.get(1, [])
                if next_agents:
                    agent_mentions = " ".join([f"@{agent.name}" for agent in next_agents])
                    followup_message = f"""Thank you for that assessment. {agent_mentions}: please build on this with your specific expertise.
                    
Be concise and focus only on how your specialty contributes to concrete next steps."""
                else:
                    # If no specific agents planned, just ask for concrete next steps
                    followup_message = "Thank you for that assessment. What are the next concrete steps we should take?"
                
                self.add_message(followup_message, self.user_name, "user")
                print(f"\n\033[1m--- Building a solution ---\033[0m\n")
            elif round_num == rounds - 1:
                # For final round, clearly specify what we need and limit further discussion
                followup_message = f"""Let's finalize our approach. I need:
                
1. A concrete action plan with 2-3 specific steps
2. Clear responsibilities for each specialist
3. Expected outcomes

Please be specific and concise. No need for additional discussion after this."""
                self.add_message(followup_message, self.user_name, "user")
                print(f"\n\033[1m--- Finalizing the plan ---\033[0m\n")
            
            # Add structured transitions between discussion rounds
            if round_num > 0:
                # Based on the discussion so far, identify a key theme to focus on
                discussion_themes = [
                    "clarifying the methodology",
                    "connecting theoretical and practical aspects",
                    "analyzing potential implications", 
                    "considering alternative approaches",
                    "identifying next experimental steps",
                    "addressing potential challenges",
                    "integrating different perspectives",
                    "evaluating evidence quality"
                ]
                
                # Select a relevant theme for this transition
                current_theme = random.choice(discussion_themes)
                
                # Create a natural transition message focused on this theme
                transition_messages = [
                    f"Let's explore {current_theme} in more depth. What insights can you share?",
                    f"I'm interested in {current_theme} now. Could you build on our earlier points?",
                    f"Let's shift our focus a bit toward {current_theme}. What are your thoughts?",
                    f"Building on what we've discussed, I'd like to hear more about {current_theme}.",
                    f"Let's look at {current_theme} from different perspectives."
                ]
                transition_message = random.choice(transition_messages)
                
                # Determine who to address in this round
                if round_num == rounds - 1:  # Final round
                    # For the final round, involve everyone who hasn't participated much
                    less_active_agents = [a for a in self.agents if a.name not in participating_agents or 
                                        list(participating_agents).count(a.name) < round_num/2]
                    if less_active_agents:
                        # Pick 2-3 less active agents to highlight
                        highlight_agents = random.sample(less_active_agents, min(3, len(less_active_agents)))
                        agent_mentions = " ".join([f"@{a.name}" for a in highlight_agents])
                        transition_message = f"{agent_mentions}: {transition_message} I'd especially value your perspectives as we finalize our discussion."
                    else:
                        # General call to the team
                        transition_message = f"Team: {transition_message} Let's bring this discussion toward a conclusion."
                else:
                    # For middle rounds, pick specific agents to involve
                    available_agents = [a for a in self.agents if a.name not in participating_agents or 
                                      list(participating_agents).count(a.name) < round_num]
                    if available_agents:
                        highlight_agents = random.sample(available_agents, min(2, len(available_agents)))
                        agent_mentions = " ".join([f"@{a.name}" for a in highlight_agents])
                        transition_message = f"{agent_mentions}: {transition_message}"
                
                # Add the transition message to the conversation
                self.add_message(transition_message, self.user_name, "user")
                print(f"\n\033[1m--- {current_theme.capitalize()} ---\033[0m\n")
            
            # Determine which agents should respond in this round
            responding_agents = []
            
            # First, get any directly addressed agents from the last user message
            directly_addressed = self.get_directly_addressed_agents()
            if directly_addressed:
                responding_agents = directly_addressed
            # Then, if we have planned participants for this round, add them
            elif round_num in participation_plan:
                responding_agents = participation_plan[round_num]
            # Otherwise, use agents mentioned in previous responses but not yet participated
            elif mentioned_agents:
                for name in mentioned_agents:
                    agent = self.get_agent_by_name(name)
                    if agent and agent.name not in participating_agents:
                        responding_agents.append(agent)
                        participating_agents.add(agent.name)
            
            # If still no one to respond, select up to 2 random agents who haven't participated
            if not responding_agents:
                available_agents = [a for a in self.agents if a.name not in participating_agents]
                if available_agents:
                    # Limit to at most 2 new agents per round to avoid explosion
                    new_agents = random.sample(available_agents, min(2, len(available_agents)))
                    responding_agents.extend(new_agents)
                    for agent in new_agents:
                        participating_agents.add(agent.name)
            
            # Process each agent's response for this round
            for agent in responding_agents:
                # Skip the starter agent in round 0 (already responded)
                if round_num == 0 and agent == starter_agent:
                    continue
                
                # Show progress indicator
                progress = ProgressIndicator(agent.name, "formulating a response")
                progress.start()
                
                try:
                    # Standard context for lab discussion
                    agent_context = f"""You are {agent.name}, speaking in a casual lab meeting with colleagues you know well. Guidelines for natural conversation:

1. Speak to your colleagues the way researchers talk when the formal presentation is over and everyone is just brainstorming
2. Don't use formal addressing patterns like "@Name:" - just refer to people naturally in your speech
3. Use shorthand, make casual references to shared knowledge, and be conversational
4. Express mild disagreement, enthusiasm, curiosity, or confusion as you naturally would
5. Use the technical language and casual phrasing that experts in {agent.specialty} use among themselves
6. Reference what others have said directly without formal citations
7. Avoid repeating what others have already said - add NEW information

Imagine this is a transcript of a real lab conversation, not a formal discussion."""
                    
                    # Generate response that builds on the conversation
                    specialized_history = self.conversation_history.copy()
                    if agent_context:
                        specialized_history.append({
                            "role": "system",
                            "content": agent_context
                        })
                    
                    if round_num == 0:
                        # For first round, call generate_response_with_research but still pass the context
                        response = agent.generate_response_with_research(topic)
                    else:
                        # Use the regular response method with conversation history
                        response = agent.generate_response(specialized_history)
                    
                    
                    # Add the final response to the conversation
                    self.add_message(response, agent.name, "assistant")
                finally:
                    progress.stop()
                
                # Format and print the response
                self._print_agent_message(agent.name, response, agent_colors, COLORS)
                
                all_responses.append((agent.name, response))
                
                # Track which new agents were mentioned in this response
                newly_mentioned = self._extract_mentioned_agents(response)
                mentioned_agents.update(newly_mentioned)
        
        print(f"\n\033[1m🔬 Agent discussion on '{topic}' completed.\033[0m\n")
        
        # Display conversation map if enabled
        if show_conversation_map and all_responses:
            # Convert all_responses format to match what _display_conversation_map expects
            formatted_responses = []
            for agent_name, response in all_responses:
                agent = self.get_agent_by_name(agent_name)
                if agent:
                    formatted_responses.append((agent, response))
            
            # Display the conversation map
            self._display_conversation_map(formatted_responses, agent_colors, COLORS)
        
        return all_responses
        
    def _extract_mentioned_agents(self, message: str) -> set:
        """
        Extract all agents mentioned in a message.
        
        Args:
            message: The message to check for mentions.
            
        Returns:
            Set of agent names mentioned in the message.
        """
        mentioned = set()
        if not message:
            return mentioned
            
        # First check for standard format with colon: @AgentName:
        for agent in self.agents:
            mention_pattern = f"@{agent.name}:"
            if mention_pattern.lower() in message.lower():
                # Verify this is a legitimate mention (at start of line or after whitespace)
                index = message.lower().find(mention_pattern.lower())
                is_valid_mention = index == 0 or message[index-1] in " \n\t"
                if is_valid_mention:
                    mentioned.add(agent.name)
        
        # If no mentions found with standard format, try looser format: @AgentName
        if not mentioned:
            for agent in self.agents:
                mention_pattern = f"@{agent.name}"
                if mention_pattern.lower() in message.lower():
                    # Verify this is a legitimate mention (at start of line or after whitespace)
                    index = message.lower().find(mention_pattern.lower())
                    pattern_len = len(mention_pattern)
                    is_valid_start = index == 0 or message[index-1] in " \n\t"
                    is_valid_end = (index + pattern_len >= len(message) or 
                                  message[index + pattern_len] in " \n\t.,;:!?")
                    
                    if is_valid_start and is_valid_end:
                        mentioned.add(agent.name)
                    
        return mentioned
        
    def _extract_technical_concepts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract technical terminology and concepts from conversation.
        
        Args:
            messages: List of conversation messages to analyze.
            
        Returns:
            List of technical terms and concepts found in the conversation.
        """
        import re
        technical_terms = set()
        
        # Patterns for identifying technical terms
        patterns = [
            r'([A-Z][a-zA-Z]+-[A-Z][a-zA-Z]+)',  # Hyphenated terms like RAG-ESM
            r'\b([A-Z]{2,}(?:-[0-9])?)\b',       # Acronyms like ESM, ESM-2
            r'\b(evotuning|fine-tuning|retrieval|inference)\b',  # Domain terms
            r'\b([a-z]+\s+model[s]?)\b',         # X model(s)
            r'\b([a-z]+\s+learning)\b',          # X learning
            r'\b(algorithm[s]?|approach(?:es)?|method[s]?)\b' # Technical action terms
        ]
        
        # Extract terms from all messages
        for message in messages:
            content = message.get("content", "")
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                # Handle both string matches and tuple matches (from regex groups)
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m and len(m) > 2:
                                technical_terms.add(m.lower())
                    elif match and len(match) > 2:
                        technical_terms.add(match.lower())
        
        return list(technical_terms)

    def _create_excerpt(self, content: str, concepts: List[str], max_length: int = 150) -> str:
        """
        Create a relevant excerpt from content highlighting concepts.
        
        Args:
            content: The text content to excerpt from.
            concepts: List of technical concepts to highlight.
            max_length: Maximum length of the excerpt.
            
        Returns:
            A relevant excerpt from the content highlighting key concepts.
        """
        import re
        # Find most relevant section containing technical concepts
        best_section = content
        
        # Look for sentences containing concepts
        sentences = re.split(r'[.!?]\s+', content)
        relevant_sentences = []
        for sentence in sentences:
            if any(concept in sentence.lower() for concept in concepts):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Join 1-3 relevant sentences
            excerpt = ". ".join(relevant_sentences[:3])
            if len(excerpt) <= max_length:
                return excerpt
            return excerpt[:max_length] + "..."
        
        # Fallback: return start of content
        return content[:max_length] + "..."
    
    def discuss_search_results(self) -> bool:
        """
        Have some agents read papers while others ask questions about them in a natural way.
        
        Returns:
            True if discussion was successful, False otherwise.
        """
        if not self.last_search_results:
            print("No search results to discuss.")
            return False
        
        # Define ANSI color codes for output
        COLORS = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        
        # Create a mapping of agents to consistent colors
        agent_colors = {}
        available_colors = list(COLORS.keys())[:-2]  # Exclude reset and bold
        for idx, agent in enumerate(self.agents):
            agent_colors[agent.name] = available_colors[idx % len(available_colors)]
        
        # First, identify the PI agent to ensure they're always a facilitator/questioner
        pi_agent = next((agent for agent in self.agents if "PI" in agent.specialty), None)
        
        # Create agent lists - excluding PI from potential readers from the start
        agent_list = [a for a in self.agents if not (pi_agent and a.name == pi_agent.name)]
        random.shuffle(agent_list)  # Randomize before splitting
        
        # Split non-PI agents between readers and questioners
        num_readers = max(1, len(agent_list) // 2)  # At least one reader
        reader_agents = agent_list[:num_readers] 
        questioner_agents = agent_list[num_readers:]
        
        # Add PI to questioners (if we have a PI)
        if pi_agent:
            questioner_agents.append(pi_agent)
        
        # If we have too few agents, adjust the balance
        if not questioner_agents and len(self.agents) > 1:
            questioner_agents = [reader_agents.pop()]  # Move one reader to questioner
        
        # Debug print to show final agent assignments
        print("\n\033[1;33m🔍 Agent roles assigned:\033[0m")
        print(f"\033[1;33m   Reader agents: {', '.join([a.name for a in reader_agents])}\033[0m")
        print(f"\033[1;33m   Questioner agents: {', '.join([a.name for a in questioner_agents])}\033[0m")
        
        # Assign papers to reader agents only - each paper only gets assigned to one agent
        print(f"\n\033[1m📚 {len(reader_agents)} agents are reviewing {len(self.last_search_results)} papers while {len(questioner_agents)} agents will ask questions...\033[0m")
        
        # Distribute papers to reader agents
        random.shuffle(self.last_search_results)  # Randomize paper order
        agent_papers = {agent.name: [] for agent in self.agents}  # Initialize for all agents
        
        # Assign each paper to exactly one reader agent
        for i, paper in enumerate(self.last_search_results):
            # Use modulo to distribute papers evenly among readers
            reader_index = i % len(reader_agents)
            agent = reader_agents[reader_index]
            agent_papers[agent.name].append(paper)
        
        # Also track papers by ID for reference during discussion
        papers_by_id = {}
        paper_summaries = {}  # Used to create context for questioner agents
        
        # Have reader agents read their assigned papers
        reading_prompts = []
        
        for agent in reader_agents:
            agent_name = agent.name
            papers = agent_papers[agent_name]
            
            if not papers:
                continue
            
            # Construct reading prompt with paper details
            paper_info = []
            paper_brief_summaries = []  # Shorter summaries for questioners
            
            for i, paper in enumerate(papers):
                title = paper.get('title', 'Untitled paper')
                url = paper.get('url', 'No URL available')
                
                # Create a unique ID for the paper (arxiv_id or url)
                paper_id = paper.get('arxiv_id') if paper.get('source') == 'arxiv' and paper.get('arxiv_id') else url
                papers_by_id[paper_id] = paper
                
                # Get the content, which might be in summary, snippet, or content field
                content = ""
                
                # Check if this paper is already in the cache
                if paper_id in self.paper_cache:
                    content = self.paper_cache[paper_id]
                    print(f"\033[1;32m✓ Using cached content for: {title[:30]}...\033[0m")
                # If not in cache, try to download it or load it from local folder
                elif ((paper.get('source') == 'arxiv' and paper.get('arxiv_id')) or 
                      (paper.get('source') == 'local_folder' and paper.get('file_path')) or 
                      paper.get('url', '').startswith('http')):
                    # Create progress indicator for PDF processing
                    message = "Reading" if paper.get('source') == 'local_folder' else "Downloading and processing"
                    pdf_progress = ProgressIndicator(None, f"{message} PDF for {title[:30]}...")
                    pdf_progress.start()
                    
                    try:
                        # Import fetch_paper_content from document_loader
                        from ..utils.document_loader import fetch_paper_content
                        
                        # Try to get the full text from PDF
                        if paper.get('source') == 'arxiv' and paper.get('arxiv_id'):
                            # Use centralized function for arXiv papers
                            full_text = fetch_paper_content(paper.get('arxiv_id'), source="arxiv")
                            if "Full Text:" in full_text:
                                content = full_text
                                # Add to cache
                                self.paper_cache[paper_id] = content
                                print(f"\033[1;32m✓ Successfully extracted full text from arXiv paper: {title[:30]}...\033[0m")
                            else:
                                # If we only got abstract, still use it and cache it
                                content = full_text
                                self.paper_cache[paper_id] = content
                        elif paper.get('source') == 'biorxiv' and paper.get('doi'):
                            # Use centralized function for bioRxiv papers
                            full_text = fetch_paper_content(paper.get('doi'), source="biorxiv")
                            if "Full Text:" in full_text:
                                content = full_text
                                # Add to cache
                                self.paper_cache[paper_id] = content
                                print(f"\033[1;32m✓ Successfully extracted full text from bioRxiv paper: {title[:30]}...\033[0m")
                            else:
                                content = full_text
                                self.paper_cache[paper_id] = content
                        elif paper.get('source') == 'local_folder' and paper.get('file_path'):
                            # Use centralized function for local papers
                            file_path = paper.get('file_path')
                            content = fetch_paper_content(file_path, source="local_folder")
                            if "Full Text:" in content:
                                # Add to cache
                                self.paper_cache[paper_id] = content
                                print(f"\033[1;32m✓ Successfully read local paper: {title[:30]}...\033[0m")
                            else:
                                # Store whatever content we got
                                self.paper_cache[paper_id] = content
                                print(f"\033[1;32m✓ Processed local paper: {title[:30]}...\033[0m")
                        elif paper.get('url'):
                            # For other papers with URLs, use the centralized function
                            url = paper.get('url')
                            full_text = fetch_paper_content(url, source="url")
                            
                            if "Error" not in full_text and len(full_text) > 200:
                                # We have substantial content, use it and cache it
                                content = full_text
                                self.paper_cache[paper_id] = content
                                print(f"\033[1;32m✓ Successfully extracted content from URL: {url[:50]}...\033[0m")
                    except Exception as e:
                        print(f"\033[91mError fetching full paper content: {str(e)}\033[0m")
                    finally:
                        pdf_progress.stop()
                
                # If full content retrieval failed, fall back to available metadata
                if not content:
                    if 'summary' in paper and paper['summary']:
                        content = paper['summary']
                    elif 'snippet' in paper and paper['snippet']:
                        content = paper['snippet']
                    elif 'content' in paper and paper['content'] and isinstance(paper['content'], str):
                        content = paper['content'][:2000]  # Increased limit for better context
                    
                    # Cache the fallback content too
                    if content and paper_id and paper_id not in self.paper_cache:
                        self.paper_cache[paper_id] = content
                
                authors = ""
                if 'authors' in paper and paper['authors']:
                    if isinstance(paper['authors'], list):
                        authors = ", ".join(paper['authors'][:5])
                        if len(paper['authors']) > 5:
                            authors += ", et al."
                    else:
                        authors = str(paper['authors'])
                
                published = paper.get('published', '')
                
                # Full paper text for readers
                paper_text = f"Paper {i+1}: {title}\n"
                if authors:
                    paper_text += f"Authors: {authors}\n"
                if published:
                    paper_text += f"Published: {published}\n"
                paper_text += f"URL: {url}\n"
                if content:
                    paper_text += f"Abstract/Content:\n{content}\n"
                
                paper_info.append(paper_text)
                
                # Create brief summary for questioner context
                brief_summary = f"- {title}"
                if authors:
                    brief_summary += f" by {authors}"
                if published:
                    brief_summary += f" (published: {published})"
                brief_summary += f" (being read by {agent_name})"
                paper_brief_summaries.append(brief_summary)
                
                # Add to paper summaries for questioners
                paper_summaries[paper_id] = {
                    "title": title,
                    "authors": authors,
                    "reader": agent_name,
                    "brief": brief_summary,
                    "paper_id": paper_id
                }
            
            paper_texts = "\n\n".join(paper_info)
            
            # Create a specialized prompt for reader agents
            paper_discussion_instructions = agent.get_communication_instructions() if hasattr(agent, 'get_communication_instructions') else ""
            
            reading_prompt = f"""You're in a lab meeting where you'll discuss papers with colleagues. You've just read these papers in detail:

{paper_texts}

Other colleagues HAVE NOT read these papers - you are the expert who will share the key insights with them. They will ask you questions to better understand the papers.

{paper_discussion_instructions}

PAPER DISCUSSION APPROACH:
1. Begin by CLEARLY LISTING EACH PAPER you've read by title and briefly what each is about
2. Share what you found most interesting or significant about EACH paper
3. Mention specific methods, results, or conclusions from the papers
4. Be prepared to answer questions from colleagues who haven't read the papers
5. Use casual, conversational language as if talking to friends
6. Refer to specific sentences, figures, or sections from the papers
7. Express genuine curiosity or skepticism about certain claims

CRITICAL: Your colleagues have not read these papers and need you to clearly introduce each one. DO NOT assume they know what papers you've read - explicitly mention each paper by title at the beginning of your response.

Remember: Your colleagues are counting on you to explain these papers accurately. Be thorough but conversational.
"""
            reading_prompts.append((agent, reading_prompt))
        
        # Generate initial analyses from reader agents about the papers
        print("\n\033[1;32m🔍 Reading papers and preparing for discussion...\033[0m")
        
        # Have reader agents share their initial thoughts
        for agent, reading_prompt in reading_prompts:
            # Show progress indicator
            progress = ProgressIndicator(agent.name, "reviewing paper")
            progress.start()
            
            try:
                # Get the papers assigned to this specific agent
                assigned_papers = agent_papers[agent.name]
                
                # Don't use paper metadata in the header for reader agents discussing multiple papers
                # Instead, they'll mention all papers naturally in their response
                paper_metadata = None
                
                # Generate the specialized analysis with paper metadata
                response = agent.generate_response_with_specialized_prompt(reading_prompt, paper_metadata)
                
                # Add to conversation history
                self.add_message(response, agent.name, "assistant")
                
                # Format and print the response
                self._print_agent_message(agent.name, response, agent_colors, COLORS)
            finally:
                progress.stop()
        
        # Now have questioner agents ask questions about the papers
        print("\n\033[1m💬 Starting interactive discussion about the papers...\033[0m")
        
        # Create a list of all paper summaries for questioners
        all_paper_summaries = "\n".join([summary["brief"] for _, summary in paper_summaries.items()])
        
        # Generate prompts for questioner agents
        for i, questioner in enumerate(questioner_agents):
            # Create a specific focus area for each questioner to avoid repetitive questions
            focus_areas = [
                "methodological aspects and potential limitations",
                "key findings and their implications",
                "connections to existing research and theories",
                "potential applications and future research directions",
                "underlying assumptions and theoretical frameworks"
            ]
            
            # Assign a focus area to this questioner
            focus = focus_areas[i % len(focus_areas)]
            
            # Special focus for PI if present
            if "PI" in questioner.specialty:
                focus = "critical evaluation and integration of findings across papers"
            
            # Create a direct question from the questioner to a reader
            reader_names = [agent.name for agent in reader_agents]
            question_to = random.choice(reader_names) if reader_names else "the team"
            
            # Show progress indicator for the questioner
            progress = ProgressIndicator(questioner.name, "formulating a question")
            progress.start()
            
            try:
                # Extract conversation history for context
                recent_messages = []
                for message in self.conversation_history:
                    if message.get("role") == "assistant":
                        msg_content = message.get("content", "")
                        msg_name = message.get("name", "")
                        # Include full message content
                        recent_messages.append(f"{msg_name}: {msg_content}")
                
                # Format the conversation history into a readable context
                conversation_context = "\n\n".join(recent_messages)
                
                # Craft a question prompt based on the focus area
                if "PI" in questioner.specialty:
                    # PI gets a facilitator role
                    question_prompt = f"""You are {questioner.name}, the PI facilitating a lab discussion about recent papers.

Papers being discussed:
{all_paper_summaries}

Your role is to guide the conversation by asking questions that help integrate findings across papers or highlight important connections.

FULL CONVERSATION HISTORY (review carefully to avoid asking duplicate questions):
{conversation_context}

INSTRUCTIONS:
1. Begin your question with "@{question_to}:" (include the @ symbol and colon)
2. Focus on {focus}
3. Be conversational and curious, as if in a real lab meeting
4. Ask about specific aspects of their papers, not just general impressions
5. Your question should encourage detailed, substantive responses
6. DO NOT repeat questions that have already been asked by other team members
7. Build on the existing conversation in a natural way

IMPORTANT: Your message MUST start with "@{question_to}:" to ensure proper conversation threading.

Frame your question in a way that will help the team draw connections between these papers and your lab's research focus."""
                else:
                    # Regular questioners ask about specific papers
                    question_prompt = f"""You are {questioner.name}, a researcher participating in a lab meeting about recent papers.

Papers being discussed:
{all_paper_summaries}

You haven't read these papers yourself, but you're curious to learn more from your colleagues who have.

FULL CONVERSATION HISTORY (review carefully to avoid asking duplicate questions):
{conversation_context}

INSTRUCTIONS:
1. Begin your question with "@{question_to}:" (include the @ symbol and colon) 
2. Focus on {focus}
3. Be conversational and curious, as if in a real lab meeting
4. Your question should be about specific aspects of their papers, not just general impressions
5. Draw from your expertise as a {questioner.specialty} when framing your question
6. DO NOT repeat questions that have already been asked by other team members
7. Build on the existing conversation in a natural way

IMPORTANT: Your message MUST start with "@{question_to}:" to ensure proper conversation threading.

Ask a question that will help you better understand these papers from your unique disciplinary perspective."""
                
                # Generate a natural question from the questioner
                question = questioner.generate_response_with_specialized_prompt(question_prompt)
                
                # Add to conversation history as coming from the questioner (not the user)
                self.add_message(question, questioner.name, "assistant")
                
                # Format and print the question
                self._print_agent_message(questioner.name, question, agent_colors, COLORS)
            finally:
                progress.stop()
            
            # Now have the addressed agent respond
            addressed_agent = self.get_agent_by_name(question_to)
            if addressed_agent:
                # Show progress indicator
                progress = ProgressIndicator(addressed_agent.name, "formulating a response")
                progress.start()
                
                try:
                    # Create a specialized context for this response
                    assigned_papers = agent_papers[addressed_agent.name]
                    paper_titles = [paper.get('title', 'Untitled paper') for paper in assigned_papers]
                    
                    # Find the specific question that was just asked
                    question_content = ""
                    for message in reversed(self.conversation_history):
                        if message.get("role") == "assistant" and message.get("name") == questioner.name:
                            question_content = message.get("content", "")
                            break
                    
                    # Create a response prompt that includes the actual question being asked
                    response_context = f"""Your colleague {questioner.name} has just asked you this specific question:

"{question_content}"

The papers you read were:
{', '.join(paper_titles)}

You're the expert on these papers since you've read them thoroughly. Your colleagues are counting on you to share accurate, detailed information in a natural conversational way.

ANSWER APPROACH:
1. Start with "@{questioner.name}:" to directly address them (make sure to include the @ symbol and colon)
2. Share specific details from the papers that answer their question
3. Use examples, data, or quotes from the papers to support your points
4. Draw connections between different sections of the papers
5. Acknowledge any limitations or gaps in the research
6. If appropriate, ask your colleague a follow-up question about their interests

IMPORTANT: Always begin your response with "@{questioner.name}:" to ensure the conversation thread is clear.

Your goal is to be both informative and conversational, sharing your expertise while maintaining a natural lab discussion flow."""
                    
                    # Generate response
                    response = addressed_agent.generate_response_with_specialized_prompt(response_context)
                    
                    # Add to conversation history
                    self.add_message(response, addressed_agent.name, "assistant")
                    
                    # Format and print the response
                    self._print_agent_message(addressed_agent.name, response, agent_colors, COLORS)
                finally:
                    progress.stop()
            
            # After the first direct Q&A, have PI or another agent facilitate broader discussion
            if i == 0:
                facilitator = pi_agent if pi_agent else random.choice(questioner_agents)
                if facilitator:
                    # Show progress indicator
                    facilitator_progress = ProgressIndicator(facilitator.name, "guiding discussion")
                    facilitator_progress.start()
                    
                    try:
                        # Get the actual list of available agent names to prevent hallucinated mentions
                        available_agents = [a.name for a in self.agents if a != facilitator]
                        agents_list = ", ".join([f"@{name}" for name in available_agents[:min(3, len(available_agents))]])
                        
                        # Extract recent conversation history to ground the facilitator
                        recent_messages = []
                        for message in self.conversation_history[-5:]:  # Last 5 messages
                            if message.get("role") == "assistant":
                                msg_content = message.get("content", "")
                                msg_name = message.get("name", "")
                                if len(msg_content) > 300:  # Truncate long messages
                                    msg_content = msg_content[:300] + "..."
                                recent_messages.append(f"{msg_name}: {msg_content}")
                        
                        # Get reader agents and what papers they've read for proper context
                        paper_context = []
                        for agent_name, papers in agent_papers.items():
                            if papers:  # Only include agents who actually read papers
                                agent_papers_titles = [paper.get('title', 'Untitled paper') for paper in papers]
                                paper_context.append(f"{agent_name} read: {', '.join(agent_papers_titles)}")
                        
                        paper_context_str = "\n".join(paper_context)
                        recent_context = "\n\n".join(recent_messages)
                        
                        facilitation_prompt = f"""You are {facilitator.name}, helping guide this lab discussion about specific research papers.

IMPORTANT CONTEXT - Papers being discussed:
{paper_context_str}

Recent conversation (REVIEW CAREFULLY before generating your response):
{recent_context}

The colleagues available to address in this discussion are: {agents_list}

INSTRUCTIONS:
1. Briefly acknowledge a specific point from the recent conversation
2. Invite 1-2 specific colleagues from the list above to share their thoughts, using the @Name format
3. ONLY ask questions directly related to papers that were actually discussed in the conversation
4. Make sure your questions are specific to the paper content mentioned by others - don't hallucinate details
5. Your questions must be directly grounded in what was already shared, not hypothetical or generic

CRITICAL RULES:
- ONLY reference topics, methods, or findings that were explicitly mentioned in the previous messages
- DO NOT invent or assume details about the papers that weren't already shared
- DO NOT mention "gene regulation," "model organisms," or research concepts that weren't brought up
- DO NOT refer to experiment design or confounding variables unless someone specifically mentioned these
- Address colleagues using only the exact names from the list above with @ symbol

Your goal is to facilitate a natural, flowing lab discussion while staying strictly anchored to what has actually been discussed.
"""
                        # Generate facilitation message
                        facilitation_message = facilitator.generate_response_with_specialized_prompt(facilitation_prompt)
                        
                        # Add to conversation history as coming from the facilitator
                        self.add_message(facilitation_message, facilitator.name, "assistant")
                        
                        # Format and print the facilitation message
                        self._print_agent_message(facilitator.name, facilitation_message, agent_colors, COLORS)
                    finally:
                        facilitator_progress.stop()
        
        # Final interactive discussion round with multiple agents responding
        num_final_responses = min(3, len(self.agents))
        self.generate_responses(num_final_responses, enable_collaboration=True)
        
        # Have the PI summarize if present, otherwise a random agent
        summarizer = pi_agent if pi_agent else random.choice(self.agents)
        
        # Create a summary prompt for wrapping up the discussion
        summary_prompt = f"""The team has had a great discussion about these papers. @{summarizer.name}: Could you summarize the key takeaways from our discussion and suggest how these papers might inform our future research?"""
        self.user_message(summary_prompt)
        
        # Generate the summary response
        if summarizer:
            # Show progress indicator
            progress = ProgressIndicator(summarizer.name, "synthesizing discussion")
            progress.start()
            
            try:
                # Get the list of all agent names for proper addressing
                all_agent_names = [a.name for a in self.agents if a != summarizer]
                agents_str = ", ".join([f"@{name}" for name in all_agent_names[:min(3, len(all_agent_names))]])
                
                # Extract information about the papers that were actually discussed
                papers_info = []
                for paper_id, paper in papers_by_id.items():
                    title = paper.get('title', 'Untitled paper')
                    authors = []
                    if 'authors' in paper and paper['authors']:
                        if isinstance(paper['authors'], list):
                            authors = paper['authors'][:3]  # First 3 authors only to keep it concise
                        else:
                            authors = [str(paper['authors'])]
                    
                    authors_str = ", ".join(authors)
                    if authors and len(authors) > 3:
                        authors_str += ", et al."
                        
                    source = paper.get('source', 'unknown source')
                    paper_info = f"- \"{title}\"" + (f" by {authors_str}" if authors_str else "") + f" ({source})"
                    papers_info.append(paper_info)
                
                papers_discussed = "\n".join(papers_info)
                
                # Extract recent conversation (last 5-10 messages) to give context about what was discussed
                recent_messages = []
                relevant_message_count = min(10, len(self.conversation_history))
                for message in self.conversation_history[-relevant_message_count:]:
                    if message.get("role") == "assistant":
                        sender = message.get("name", "Unknown")
                        content = message.get("content", "")
                        
                        # Truncate long messages to keep the context manageable
                        if len(content) > 300:
                            content = content[:300] + "..."
                            
                        recent_messages.append(f"{sender}: {content}")
                
                conversation_excerpt = "\n\n".join(recent_messages)
                
                summary_context = f"""The lab has been having a rich discussion about the following research papers:

{papers_discussed}

Recent conversation excerpts:
{conversation_excerpt}

As we wrap up, you should synthesize the key points from the discussion.

Your colleagues in this discussion are: {agents_str}

SYNTHESIS APPROACH:
1. Acknowledge the SPECIFIC papers that were discussed (use their actual titles)
2. Highlight 2-3 key insights that emerged from the discussion about these papers
3. Note any contrasting viewpoints or complementary ideas that were mentioned
4. Suggest 1-2 concrete ways these specific papers could inform your lab's future work
5. End with an encouraging note about the value of the discussion

IMPORTANT: 
- Only reference the papers and topics that were actually listed above
- NEVER mention neural networks, computer vision, or medical imaging unless they were specifically discussed
- Address the group as a whole rather than individual colleagues
- You can acknowledge contributions by using "@Name" format, but only use names from the list above
- Do not invent or mention people who are not on the list
- Keep your tone conversational while providing substantive closure to the discussion"""
                
                # Generate summary
                response = summarizer.generate_response_with_specialized_prompt(summary_context)
                
                # Add to conversation history
                self.add_message(response, summarizer.name, "assistant")
                
                # Format and print the response
                self._print_agent_message(summarizer.name, response, agent_colors, COLORS)
            finally:
                progress.stop()
        
        return True

    def _print_agent_message(self, agent_name: str, message: str, agent_colors: Dict[str, str], colors: Dict[str, str]) -> None:
        """
        Print an agent's message with formatting, highlighting direct mentions of other agents.
        
        Args:
            agent_name: The name of the agent.
            message: The message content.
            agent_colors: Mapping of agent names to color keys.
            colors: Dictionary of ANSI color codes.
        """
        # Skip messages that indicate they were for someone else
        if message.startswith("[This message was for"):
            return
            
        color = colors[agent_colors[agent_name]]
        reset = colors["reset"]
        bold = colors["bold"]
        
        # Format multiline messages
        lines = message.split('\n')
        width = 110  # Width of the box (increased for better readability)
        
        # Print header with agent name
        print(f"{color}┌─── {bold}{agent_name}{reset}{color} {'─' * (width - len(agent_name) - 6)}┐{reset}")
        
        # Detect direct mentions of other agents to highlight them
        direct_mentions = {}
        for other_agent_name, other_agent_color in agent_colors.items():
            if other_agent_name != agent_name:
                # First check standard format with colon: @AgentName:
                if f"@{other_agent_name}:".lower() in message.lower():
                    direct_mentions[other_agent_name] = colors[other_agent_color]
                # Also check for mentions without colon: @AgentName
                elif f"@{other_agent_name}".lower() in message.lower():
                    # Verify it's a proper mention and not part of another word
                    mention_pattern = f"@{other_agent_name}"
                    index = message.lower().find(mention_pattern.lower())
                    pattern_len = len(mention_pattern)
                    is_valid_start = index == 0 or message[index-1] in " \n\t"
                    is_valid_end = (index + pattern_len >= len(message) or 
                                   message[index + pattern_len] in " \n\t.,;:!?")
                    
                    if is_valid_start and is_valid_end:
                        direct_mentions[other_agent_name] = colors[other_agent_color]
        
        # Print message content with word wrap and highlighted mentions
        for line in lines:
            while line:
                if len(line) <= width - 4:  # -4 for margins
                    # Check for direct mentions and highlight them
                    highlighted_line = line
                    for mentioned_agent, mention_color in direct_mentions.items():
                        # Check for different forms of mentions and highlight them
                        # First, standard format with colon
                        mention_pattern = f"@{mentioned_agent}:"
                        if mention_pattern.lower() in highlighted_line.lower():
                            # Find all occurrences with case-insensitive search
                            pattern_len = len(mention_pattern)
                            pos = 0
                            while True:
                                pos = highlighted_line.lower().find(mention_pattern.lower(), pos)
                                if pos == -1:
                                    break
                                # Extract actual text as it appears in the original
                                actual_text = highlighted_line[pos:pos+pattern_len]
                                # Replace with highlighted version
                                highlighted_line = highlighted_line.replace(
                                    actual_text, 
                                    f"{reset}{mention_color}{bold}@{mentioned_agent}{reset}{mention_color}:{reset}{color}"
                                )
                                pos += pattern_len
                        
                        # Also check for mentions without colon
                        mention_pattern = f"@{mentioned_agent}"
                        if mention_pattern.lower() in highlighted_line.lower():
                            # Check if it's a proper mention (not already handled with colon format)
                            # and not part of another word
                            pattern_len = len(mention_pattern)
                            pos = 0
                            while True:
                                pos = highlighted_line.lower().find(mention_pattern.lower(), pos)
                                if pos == -1:
                                    break
                                # Extract actual text as it appears in the original
                                actual_text = highlighted_line[pos:pos+pattern_len]
                                # Verify it's a proper mention with appropriate boundaries
                                is_valid_start = pos == 0 or highlighted_line[pos-1] in " \n\t"
                                next_pos = pos + pattern_len
                                is_valid_end = (next_pos >= len(highlighted_line) or 
                                               highlighted_line[next_pos] in " \n\t.,;:!?")
                                
                                if is_valid_start and is_valid_end:
                                    # Replace with highlighted version
                                    highlighted_line = highlighted_line.replace(
                                        actual_text, 
                                        f"{reset}{mention_color}{bold}{actual_text}{reset}{color}"
                                    )
                                pos += 1
                    
                    print(f"{color}│ {highlighted_line}{reset}")
                    line = ""
                else:
                    # Find a good breaking point
                    break_point = width - 4
                    while break_point > 0 and line[break_point] != ' ':
                        break_point -= 1
                    if break_point == 0:  # No space found, hard break
                        break_point = width - 4
                    
                    # Get the current line segment and highlight any mentions
                    line_segment = line[:break_point]
                    for mentioned_agent, mention_color in direct_mentions.items():
                        mention_pattern = f"@{mentioned_agent}:"
                        if mention_pattern in line_segment:
                            # Highlight the mention
                            line_segment = line_segment.replace(
                                mention_pattern, 
                                f"{reset}{mention_color}{bold}@{mentioned_agent}{reset}{mention_color}:{reset}{color}"
                            )
                    
                    print(f"{color}│ {line_segment}{reset}")
                    line = line[break_point:].lstrip()
        
        # Check if message contains direct messages and show an indicator
        direct_message_indicators = []
        for other_agent_name in agent_colors:
            if other_agent_name != agent_name:
                # Check standard format with colon: @AgentName:
                if f"@{other_agent_name}:".lower() in message.lower():
                    # Verify this is a real mention (at beginning of line or after whitespace)
                    index = message.lower().find(f"@{other_agent_name}:".lower())
                    is_valid_mention = index == 0 or (index > 0 and message[index-1] in " \n\t")
                    
                    if is_valid_mention:
                        mention_color = colors[agent_colors[other_agent_name]]
                        direct_message_indicators.append(f"{mention_color}{other_agent_name}{reset}")
                
                # Also check looser format without colon: @AgentName
                elif f"@{other_agent_name}".lower() in message.lower():
                    # Verify this is a real mention (at beginning of line or after whitespace)
                    mention_pattern = f"@{other_agent_name}"
                    index = message.lower().find(mention_pattern.lower())
                    pattern_len = len(mention_pattern)
                    is_valid_start = index == 0 or message[index-1] in " \n\t"
                    is_valid_end = (index + pattern_len >= len(message) or 
                                   message[index + pattern_len] in " \n\t.,;:!?")
                    
                    if is_valid_start and is_valid_end:
                        mention_color = colors[agent_colors[other_agent_name]]
                        direct_message_indicators.append(f"{mention_color}{other_agent_name}{reset}")
        
        # Print footer with indicators of who the message is directed to
        if direct_message_indicators:
            if len(direct_message_indicators) == 1:
                direct_to = f"To: {direct_message_indicators[0]}"
            else:
                direct_to = "To: " + ", ".join(direct_message_indicators)
            print(f"{color}└{'─' * (width - len(direct_to) - 3)}[ {direct_to} {color}]─┘{reset}")
        else:
            print(f"{color}└{'─' * width}┘{reset}")
        
        print()  # Add spacing between messages