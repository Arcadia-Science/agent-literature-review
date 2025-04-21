"""
Base agent module for AI laboratory.

This code was developed with the assistance of Claude Code.
"""
from typing import Dict, List, Optional, Any
from anthropic import Anthropic
import time
import re
import random
import tiktoken

# Import tool modules
from ..tools.web_search import ArxivSearch, BioRxivSearch
from ..tools.file_operations import (
    safe_read_file, get_file_info
)

# Import role definitions
from .role_definitions import (
    PROHIBITED_PATTERNS, SPECIALTY_QUESTIONS, THIRD_PARTY_QUESTIONS, 
    NATURAL_SEGUES, TECHNICAL_INDICATORS, get_communication_instructions,
    create_pi_system_message, create_researcher_system_message,
    create_direct_response_prompt, create_research_response_system_prompt
)


class Agent:
    """Base agent class for AI laboratory."""
    
    # Track token usage across all agents
    TOKENS_IN = {}  # {model_name: token_count}
    TOKENS_OUT = {}  # {model_name: token_count}
    
    # Cost per 1K tokens for various models (updated 2025 rates)
    MODEL_COST_PER_1K = {
        # Anthropic models
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-2.1": {"input": 0.008, "output": 0.024},
        "claude-2.0": {"input": 0.008, "output": 0.024},
        # Generic fallback
        "default": {"input": 0.01, "output": 0.03}
    }
    
    def __init__(
        self, 
        name: str, 
        specialty: str, 
        description: str, 
        api_key: str,
        model: str = "claude-3-5-sonnet-20240620"
    ):
        """
        Initialize an AI agent.
        
        Args:
            name: The agent's name.
            specialty: The agent's specialty area.
            description: Detailed description of the agent's expertise.
            api_key: Anthropic API key.
            model: The model name to use, defaults to claude-3-5-sonnet-20240620.
        """
        self.name = name
        self.specialty = specialty
        self.description = description
        self.model = model
        
        # Set up Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Initialize conversation history
        self.conversation: List[Dict[str, str]] = []
        
        # Initialize agent network for cross-communication
        self.agent_network = {}
        
        # Track direct messages to/from other agents
        self.direct_messages = {}
        
        # List of all known agents in the laboratory
        self.known_agents = []
        
        # Create system message for the agent
        self.system_message = self._create_system_message()
        
        # Set user agent for API requests
        self.headers = {
            'User-Agent': 'ResearchLab/1.0 (mailto:research@example.com)'
        }
        
        # Initialize token counters for this model if not already tracking
        if model not in Agent.TOKENS_IN:
            Agent.TOKENS_IN[model] = 0
            Agent.TOKENS_OUT[model] = 0
    
    def _create_system_message(self) -> Dict[str, str]:
        """Create the system message that defines the agent's personality."""
        # Import the role definition functions
        from .role_definitions import (
            create_pi_system_message, 
            create_researcher_system_message,
            get_communication_instructions
        )
        
        # Create a formatted string of known agents with their specialties
        known_agents_str = ""
        if self.known_agents:
            known_agents_list = []
            for agent in self.known_agents:
                if agent["name"] != self.name:  # Don't include self
                    known_agents_list.append(f"- {agent['name']} ({agent['specialty']})")
            known_agents_str = "\n".join(known_agents_list)
        
        # Get communication instructions
        communication_instructions = get_communication_instructions(self.specialty)
        
        # Special system message for the PI role to drive productive outcomes
        if "PI" in self.specialty:
            content = create_pi_system_message(
                name=self.name,
                description=self.description,
                known_agents_str=known_agents_str,
                communication_instructions=communication_instructions
            )
            
            return {
                "role": "system",
                "content": content
            }
        else:
            # Regular researcher role
            content = create_researcher_system_message(
                name=self.name,
                specialty=self.specialty,
                known_agents_str=known_agents_str,
                communication_instructions=communication_instructions
            )
            
            return {
                "role": "system",
                "content": content
            }
    
    def add_to_conversation(self, role: str, content: str, name: Optional[str] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user, assistant, system).
            content: The message content.
            name: The name of the sender (for user or assistant roles).
        """
        message = {"role": role, "content": content}
        if name and role in ["user", "assistant"]:
            message["name"] = name
        
        self.conversation.append(message)
        
        # Process direct messages using @AgentName syntax if this is a message from another agent
        if role == "assistant" and name and name != self.name:
            self._process_direct_messages(content, name)
    
    def _process_direct_messages(self, content: str, sender_name: str) -> None:
        """
        Process content for direct messages using @AgentName syntax.
        
        Args:
            content: The message content to process.
            sender_name: The name of the message sender.
        """
        is_valid_mention = False
        start_idx = 0
        
        # First check standard format with colon: @AgentName:
        mention_pattern = f"@{self.name}:"
        if mention_pattern.lower() in content.lower():
            # Verify this is a legitimate mention (at start of line or after whitespace)
            index = content.lower().find(mention_pattern.lower())
            is_valid_mention = index == 0 or content[index-1] in " \n\t"
            
            if is_valid_mention:
                start_idx = index + len(mention_pattern)
        
        # If not found with colon, try looser format: @AgentName
        if not is_valid_mention:
            mention_pattern = f"@{self.name}"
            if mention_pattern.lower() in content.lower():
                # Verify it's a proper mention and not part of another word
                index = content.lower().find(mention_pattern.lower())
                pattern_len = len(mention_pattern)
                is_valid_start = index == 0 or content[index-1] in " \n\t"
                is_valid_end = (index + pattern_len >= len(content) or 
                              content[index + pattern_len] in " \n\t.,;:!?")
                
                if is_valid_start and is_valid_end:
                    is_valid_mention = True
                    start_idx = index + len(mention_pattern)
                    # Skip any punctuation after the mention
                    while start_idx < len(content) and content[start_idx] in " .,;:!?":
                        start_idx += 1
        
        if not is_valid_mention:
            return  # Not a valid direct mention
        
        # Find the end of this direct message (until next @ mention or end of message)
        end_idx = len(content)
        for agent_name in self.known_agents:
            if agent_name["name"] != self.name:  # Don't include self
                # Check for next mention with colon
                next_mention = f"@{agent_name['name']}:"
                next_idx = content.lower().find(next_mention.lower(), start_idx)
                if next_idx != -1 and next_idx < end_idx:
                    end_idx = next_idx
                
                # Also check for next mention without colon
                next_mention = f"@{agent_name['name']}"
                next_idx = content.lower().find(next_mention.lower(), start_idx)
                if next_idx != -1 and next_idx < end_idx:
                    # Verify it's a valid mention
                    pattern_len = len(next_mention)
                    is_valid_end = (next_idx + pattern_len >= len(content) or 
                                  content[next_idx + pattern_len] in " \n\t.,;:!?")
                    if is_valid_end:
                        end_idx = next_idx
        
        direct_msg = content[start_idx:end_idx].strip()
        
        # Add to direct messages from this sender
        if sender_name not in self.direct_messages:
            self.direct_messages[sender_name] = []
        
        # Track message type (question, comment, etc.) for better follow-up responses
        message_type = "general"
        if "?" in direct_msg:
            message_type = "question"
        elif any(word in direct_msg.lower() for word in ["what do you think", "opinion", "thoughts", "agree", "disagree"]):
            message_type = "opinion"
        elif any(word in direct_msg.lower() for word in ["can you", "could you", "would you", "please"]):
            message_type = "request"
            
            
        # Create the message data object
        message_data = {
            "content": direct_msg,
            "timestamp": time.time(),
            "message_type": message_type,
            "thread_id": f"{sender_name}_{len(self.direct_messages.get(sender_name, []))}"
        }
        
        
        # Add the message to direct messages
        self.direct_messages[sender_name].append(message_data)
    
    def send_direct_message(self, recipient_name: str, message: str) -> None:
        """
        Send a direct message to another agent.
        
        Args:
            recipient_name: The name of the recipient agent.
            message: The message content.
        """
        if recipient_name in self.agent_network:
            # Format the direct message with @AgentName syntax
            formatted_message = f"@{recipient_name}: {message}"
            
            # Store in our own record of sent messages
            if recipient_name not in self.direct_messages:
                self.direct_messages[recipient_name] = []
            
            self.direct_messages[recipient_name].append({
                "content": message,
                "timestamp": time.time(),
                "sent_by_me": True
            })
            
            return formatted_message
        else:
            return None
            
    def connect_to_agent(self, agent: 'Agent') -> None:
        """
        Connect this agent to another agent for direct messaging.
        
        Args:
            agent: The agent to connect with.
        """
        if agent.name != self.name:
            self.agent_network[agent.name] = agent
    
    def count_tokens(self, text_or_messages):
        """
        Count the number of tokens in text or a list of messages.
        
        Args:
            text_or_messages: Single string or list of message dictionaries.
            
        Returns:
            Number of tokens.
        """
        try:
            # Claude models all use cl100k_base tokenizer
            encoding_name = "cl100k_base"
            
            # Get the encoding
            encoding = tiktoken.get_encoding(encoding_name)
            
            # Handle different input types
            if isinstance(text_or_messages, str):
                # Simple string
                return len(encoding.encode(text_or_messages))
            elif isinstance(text_or_messages, list):
                # List of message dictionaries
                total_tokens = 0
                for message in text_or_messages:
                    if isinstance(message, dict) and "content" in message:
                        total_tokens += len(encoding.encode(message.get("content", "")))
                
                # Add approximate overhead for message formatting
                # Claude's API adds overhead for each message
                if len(text_or_messages) > 0:
                    total_tokens += 3 * len(text_or_messages)
                    
                return total_tokens
            else:
                return 0
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            return 0
    
    def update_token_usage(self, input_tokens, output_tokens):
        """
        Update token usage for cost tracking.
        
        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        if self.model in Agent.TOKENS_IN:
            Agent.TOKENS_IN[self.model] += input_tokens
            Agent.TOKENS_OUT[self.model] += output_tokens
        else:
            # Initialize if not already tracking
            Agent.TOKENS_IN[self.model] = input_tokens
            Agent.TOKENS_OUT[self.model] = output_tokens
    
    @classmethod
    def get_cost_estimate(cls):
        """
        Calculate the total cost estimate for all token usage.
        
        Returns:
            Dictionary with cost breakdown by model and total cost.
        """
        total_cost = 0
        cost_breakdown = {}
        
        for model in cls.TOKENS_IN:
            input_tokens = cls.TOKENS_IN.get(model, 0)
            output_tokens = cls.TOKENS_OUT.get(model, 0)
            
            # Get model-specific costs or use default
            if model in cls.MODEL_COST_PER_1K:
                input_cost_per_1k = cls.MODEL_COST_PER_1K[model]["input"]
                output_cost_per_1k = cls.MODEL_COST_PER_1K[model]["output"]
            else:
                # Use default pricing
                input_cost_per_1k = cls.MODEL_COST_PER_1K["default"]["input"]
                output_cost_per_1k = cls.MODEL_COST_PER_1K["default"]["output"]
            
            # Calculate costs
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            model_cost = input_cost + output_cost
            
            # Add to total and breakdown
            total_cost += model_cost
            cost_breakdown[model] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": model_cost
            }
        
        return {
            "cost_breakdown": cost_breakdown,
            "total_cost": total_cost
        }
    
    def get_communication_instructions(self) -> str:
        """
        Get communication instructions specific to this agent's role.
        These centralize guidance on how agents should communicate.
        
        Returns:
            String containing communication instructions.
        """
        # Use the centralized function from role_definitions
        from .role_definitions import get_communication_instructions
        return get_communication_instructions(self.specialty)
    
    def apply_prohibited_phrase_check(self, response: str) -> str:
        """
        Check if response contains any prohibited phrases and remove them.
        
        Args:
            response: The generated response text.
            
        Returns:
            The response with prohibited phrases removed or transformed.
        """
        # Import the prohibited patterns from role_definitions
        from .role_definitions import PROHIBITED_PATTERNS
        
        # Generate prohibited pattern strings with the agent's actual specialty
        prohibited_patterns = []
        for pattern in PROHIBITED_PATTERNS:
            # Create a regex pattern that's case-insensitive
            prohibited_patterns.append(
                re.compile(pattern.format(specialty=self.specialty), re.IGNORECASE)
            )
        
        # Check for each prohibited pattern
        for pattern in prohibited_patterns:
            # Find all matches
            matches = pattern.finditer(response)
            
            # Process matches in reverse order to avoid index issues when replacing
            matches = list(matches)
            for match in reversed(matches):
                start, end = match.span()
                
                # Check if this is at the beginning of a sentence or paragraph
                is_beginning = (start == 0 or response[start-1] in ".\n")
                
                if is_beginning:
                    # If at the beginning, just remove the phrase entirely
                    response = response[:start] + response[end:]
                else:
                    # If in the middle, replace with a neutral connector
                    response = response[:start] + "I think" + response[end:]
        
        return response
    
    def generate_response(self, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response from the agent based on the conversation history.
        
        Args:
            conversation_history: Optional override for conversation history.
            
        Returns:
            The agent's response.
        """
        # Use provided conversation history or default to self.conversation
        history = conversation_history if conversation_history else self.conversation
        
        # First check if the most recent message is from a user and explicitly addressed to specific agents
        should_stay_silent = False
        agents_str = ""
        
        if history and len(history) > 0:
            last_message = history[-1]
            if last_message.get("role") == "user":
                # Check if the user is addressing specific agents using @ notation
                content = last_message.get("content", "")
                mentioned_agents = []
                is_direct_addressal = False
                
                # First check standard format with colon: @AgentName:
                for agent in self.known_agents:
                    agent_name = agent["name"]
                    mention_pattern = f"@{agent_name}:"
                    if mention_pattern.lower() in content.lower():
                        index = content.lower().find(mention_pattern.lower())
                        # Verify it's a proper mention (at start of line or after whitespace)
                        if index == 0 or (index > 0 and content[index-1] in " \n\t"):
                            mentioned_agents.append(agent_name)
                            is_direct_addressal = True
                
                # If no mentions found with colon, try looser format: @AgentName
                if not mentioned_agents:
                    for agent in self.known_agents:
                        agent_name = agent["name"]
                        mention_pattern = f"@{agent_name}"
                        if mention_pattern.lower() in content.lower():
                            # Verify it's a proper mention and not part of another word
                            index = content.lower().find(mention_pattern.lower())
                            pattern_len = len(mention_pattern)
                            is_valid_start = index == 0 or content[index-1] in " \n\t"
                            is_valid_end = (index + pattern_len >= len(content) or 
                                            content[index + pattern_len] in " \n\t.,;:!?")
                            
                            if is_valid_start and is_valid_end:
                                mentioned_agents.append(agent_name)
                                is_direct_addressal = True
                
                # If user is using direct addressing and this agent isn't mentioned, stay silent
                if is_direct_addressal and self.name not in mentioned_agents:
                    agents_str = ", ".join(mentioned_agents)
                    should_stay_silent = True
        
        # If we determined we should stay silent, return early with a special message
        if should_stay_silent:
            return f"[This message was for {agents_str}]"  # This will be filtered out later
        
        # Extract recent agent responses to check for repetition and inform our response
        recent_agent_responses = []
        previous_ideas = set()
        previous_suggestions = []
        
        # Loop through recent history in reverse (most recent first)
        # Only consider the last 10 messages to keep it relevant
        for message in reversed(history[-10:]):
            if message.get("role") == "assistant" and message.get("name") != self.name:
                agent_name = message.get("name", "")
                message_content = message.get("content", "")
                
                # Only add substantive messages
                if len(message_content) > 50:
                    recent_agent_responses.append({
                        "name": agent_name,
                        "content": message_content
                    })
                    
                    # Extract key phrases, terms and suggestions to avoid repetition
                    lines = message_content.split('\n')
                    for line in lines:
                        # Look for suggestions in numbered or bulleted lists
                        if re.search(r'^\s*\d+\.\s', line) or re.search(r'^\s*[\*\-•]\s', line):
                            # Clean and normalize the suggestion
                            suggestion = re.sub(r'^\s*[\d\*\-•]+\.\s', '', line).strip().lower()
                            if len(suggestion) > 10:  # Only consider substantive suggestions
                                previous_suggestions.append(suggestion)
                        
                        # Extract key technical terms
                        words = line.split()
                        for i, word in enumerate(words):
                            # Look for technique names, methods, etc.
                            if (len(word) > 5 and 
                                not word.lower() in ["should", "would", "could", "about", "these", "those", "their"]):
                                previous_ideas.add(word.lower())
                            
                            # Look for multi-word technical terms
                            if i < len(words) - 1:
                                phrase = f"{word} {words[i+1]}"
                                clean_phrase = phrase.lower()
                                if (len(clean_phrase) > 10 and 
                                    not all(w in ["should", "would", "could", "about", "these", "those", "their"] for w in clean_phrase.split())):
                                    previous_ideas.add(clean_phrase)
        
        # Add the centralized communication requirements
        communication_instructions = self.get_communication_instructions()
        
        # Create a more streamlined base system prompt that emphasizes natural communication
        base_system_prompt = f"""You are {self.name}, a researcher specializing in {self.specialty}.

{communication_instructions}

CORE REQUIREMENTS:
- Build on previous messages with ORIGINAL contributions
- Add your unique expertise without labeling your specialty
- Challenge or extend at least one previous point
- Be substantive and technical but conversational
- Ask thoughtful follow-up questions when appropriate
"""
        
        # Add information about what others have already said
        if recent_agent_responses:
            # Create a summary of what's already been discussed
            recent_ideas_summary = "Recent key points from others:\n"
            
            for response in recent_agent_responses[:2]:  # Focus on last two responses
                recent_ideas_summary += f"- {response['name']}: "
                
                # Extract key points (either from first 100 chars or first bullet points)
                content = response["content"]
                bullet_points = re.findall(r'(?:^|\n)\s*[\d\*\-•]+\.\s*(.*?)(?=\n|$)', content)
                
                if bullet_points:
                    # If we found bullet points, use the first 1-2
                    points = bullet_points[:min(2, len(bullet_points))]
                    recent_ideas_summary += ", ".join(points)
                else:
                    # Otherwise just summarize the first part of the message
                    summary = content[:100].strip()
                    if len(content) > 100:
                        summary += "..."
                    recent_ideas_summary += summary
                
                recent_ideas_summary += "\n"
            
            base_system_prompt += f"\n\n{recent_ideas_summary}"
        
        # Convert conversation history to Anthropic format
        anthropic_messages = []
        
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            name = message.get("name", "")
            
            if role == "system":
                # Skip system messages as we'll use our custom system prompt
                continue
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # If message has a name and it's not from this agent, format as user message with name prefix
                if name and name != self.name:
                    anthropic_messages.append({"role": "assistant", "content": f"{name}: {content}"})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
        
        # Count input tokens for cost tracking
        input_tokens = self.count_tokens(base_system_prompt) + self.count_tokens(anthropic_messages)
        
        # Try to get a response with retry for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Use Anthropic's messages API format
                response = self.client.messages.create(
                    model=self.model,
                    system=base_system_prompt,
                    messages=anthropic_messages,
                    max_tokens=4000,
                    temperature=0.8  # Slightly higher for more diverse responses
                )
                
                # Get response content - safely handle empty responses
                try:
                    response_text = response.content[0].text
                    
                    # Check for prohibited phrases and fix them
                    response_text = self.apply_prohibited_phrase_check(response_text)
                    
                    # Update token usage for cost tracking
                    output_tokens = self.count_tokens(response_text)
                    self.update_token_usage(input_tokens, output_tokens)
                except (IndexError, AttributeError) as e:
                    # This can happen when the API returns an empty response
                    # Often occurs with very simple or contextless messages
                    print(f"Warning: Empty or invalid model response for agent {self.name}.")
                    # Fall back to a default response
                    response_text = "I'm not sure how to respond to that. Could you please provide more context or ask a specific question?"
                    
                    # Log minimal tokens for cost tracking
                    self.update_token_usage(input_tokens, self.count_tokens(response_text))
                
                # Only add agent mentions organically when appropriate
                # Check if the response already contains any @mention to another agent
                has_mention = any(f"@{agent['name']}" in response_text for agent in self.known_agents if agent['name'] != self.name)
                
                # Add a mention only if: 
                # 1. Response is substantive (not too short)
                # 2. Response is asking a question (contains a question mark)
                # 3. Response doesn't already mention someone else
                # 4. Only 40% chance to add a mention (make it feel natural, not forced)
                should_add_mention = (not has_mention and 
                                     len(response_text.strip()) > 150 and 
                                     "?" in response_text and
                                     random.random() < 0.4)
                
                if should_add_mention:
                    # Try to find another agent that isn't us and hasn't recently spoken
                    recent_speakers = [r["name"] for r in recent_agent_responses[:2]]
                    available_agents = [a for a in self.known_agents 
                                      if a['name'] != self.name and a['name'] not in recent_speakers]
                    
                    # If no other agents available, consider all agents except self
                    if not available_agents:
                        available_agents = [a for a in self.known_agents if a['name'] != self.name]
                    
                    # Make sure we have at least one agent to mention
                    if available_agents:
                        # Choose a random agent to mention
                        other_agent = random.choice(available_agents)
                        other_name = other_agent['name']
                        other_specialty = other_agent['specialty']
                        
                        # Extract key topics from our response that weren't in previous messages
                        response_lines = response_text.strip().split("\n")
                        key_topics = []
                        
                        # Look for substantive sentences with unique content
                        for line in response_lines:
                            # Skip very short lines or obvious greeting lines
                            if len(line) < 20 or "hello" in line.lower() or "thank" in line.lower():
                                continue
                                
                            # Check if this line contains content not in previous messages
                            line_lower = line.lower()
                            if not any(prev.lower() in line_lower for prev in previous_suggestions):
                                words = line.split()
                                if len(words) > 5:
                                    # Extract a relevant segment
                                    segment = " ".join(words[:8])
                                    if len(segment) > 20:
                                        key_topics.append(segment)
                            
                        # If we couldn't find unique topics, use our specialty
                        if not key_topics:
                            key_topics = [f"this aspect of the problem"]
                        
                        # Choose a topic to include in our question
                        topic_text = random.choice(key_topics)
                        
                        # Create a specialty-appropriate question for the other agent
                        specialty_questions = {
                            "Biologist": [
                                f"how would you approach this from a molecular biology perspective?",
                                f"do you see any biological constraints we should consider here?",
                                f"how might evolutionary principles apply to this problem?"
                            ],
                            "Mathematician": [
                                f"could you help formalize this reasoning with a mathematical framework?",
                                f"do you see a way to prove whether this approach is optimal?",
                                f"what statistical considerations should we keep in mind?"
                            ],
                            "Physicist": [
                                f"how do fundamental physical principles apply to this problem?",
                                f"are there any physics-based models we could adapt here?",
                                f"what conservation laws might be relevant to this system?"
                            ],
                            "Computer Scientist": [
                                f"how would you implement this algorithmically?",
                                f"are there computational optimizations we're overlooking?",
                                f"what data structures would be most efficient for this approach?"
                            ],
                            "PI": [
                                f"what do you think about this direction for the project?",
                                f"how does this fit with the lab's broader research goals?",
                                f"what experimental design would you recommend to test this?"
                            ]
                        }
                        
                        # Get questions for this specialty
                        specialty_specific = specialty_questions.get(
                            other_specialty, 
                            [f"how would your expertise in {other_specialty} apply to this?"]
                        )
                        
                        # Choose a question appropriate for their specialty
                        question = random.choice(specialty_specific)
                        
                        # Add a natural segue with the question
                        natural_segues = [
                            f"\n\n{other_name}, regarding {topic_text}... {question}",
                            f"\n\nI'd be curious to hear {other_name}'s thoughts on {topic_text}. {question}",
                            f"\n\n{other_name}, this connects to your work, doesn't it? {question}",
                            f"\n\nActually, {other_name} might have some insights here. {question}"
                        ]
                        follow_up = random.choice(natural_segues)
                        
                        # Add the follow-up to the response, but check it doesn't duplicate existing mentions
                        if f"@{other_name}" not in response_text and other_name not in response_text[-100:]:
                            response_text += follow_up
                
                return response_text
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                # Rate limit handling
                if 'rate_limit' in str(e).lower() or 'quota' in str(e).lower():
                    wait_time = 20
                    print(f"Rate limit reached. Waiting {wait_time} seconds before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use exponential backoff
                    time.sleep(2 ** attempt)
        
        # This should not be reached due to the exception in the loop
        return "I'm having trouble connecting. Please try again later."
        
    def generate_response_to_direct_message(self, sender_name: str, context: Optional[str] = None) -> str:
        """
        Generate a response to a direct message from another agent.
        
        Args:
            sender_name: The name of the agent who sent the message.
            context: Optional additional context to include.
            
        Returns:
            The agent's response to the direct message.
        """
        if sender_name not in self.direct_messages or not self.direct_messages[sender_name]:
            return f"I don't have any messages from {sender_name} to respond to."
        
        # Get the most recent message from this sender
        latest_msg = self.direct_messages[sender_name][-1]
        
        # Get list of other agent names to suggest
        other_agent_options = []
        for agent in self.known_agents:
            if agent["name"] != self.name and agent["name"] != sender_name:
                other_agent_options.append(f"{agent['name']} ({agent['specialty']})")

        # Add the centralized communication instructions
        communication_instructions = self.get_communication_instructions()
        
        # Check if we have context - this would be from the Lab's enhanced system with topic continuity
        enhanced_prompt = None
        if context:
            # Use the provided context-aware prompt from the lab
            enhanced_prompt = context
        else:
            # Create our own context for basic continuity
            # Get some of the recent conversation history for context
            recent_conversation = []
            max_context_items = 5
            context_count = 0
            
            for message in reversed(self.conversation):
                if context_count >= max_context_items:
                    break
                    
                role = message.get("role", "")
                content = message.get("content", "")
                name = message.get("name", "")
                
                # Only include user or assistant messages with names
                if role in ["user", "assistant"] and name:
                    # Only include substantive messages
                    if len(content) > 50:
                        # Include the full message from sender for direct context
                        if name == sender_name:
                            recent_conversation.append(f"{name}: {content}")
                            context_count += 1
                        else:
                            # For other messages, include a truncated version
                            if len(content) > 200:
                                content = content[:200] + "..."
                            recent_conversation.append(f"{name}: {content}")
                            context_count += 1
            
            # Reverse to maintain chronological order
            recent_conversation.reverse()
            conversation_context = "\n\n".join(recent_conversation)
        
        # Create a more natural dialogue instruction for conversational scientific communication
        direct_dialogue_prompt = {
            "role": "system",
            "content": enhanced_prompt or f"""You are {self.name}, responding to a colleague in a lab conversation.

{communication_instructions}

RECENT CONVERSATION CONTEXT:
{conversation_context}

DIRECT RESPONSE GUIDANCE:
1. Begin with "@{sender_name}:" to continue the conversation thread
2. Acknowledge their specific point or question directly
3. Provide a substantive, detailed response drawing on your expertise
4. Keep your response conversational but scientifically rigorous
5. When appropriate, include a follow-up question

SCIENTIFIC INTERACTION MODEL:
1. Analyze claims using evidence-based reasoning
2. Identify methodological limitations or alternative interpretations
3. Reference specific techniques or approaches from your field
4. Frame critique constructively while maintaining scientific rigor

If appropriate, bring another colleague into the conversation: {', '.join(other_agent_options) if other_agent_options else "No other colleagues available"}
"""
        }
        
        # Create a focused conversation for just this exchange
        focused_conversation = [
            {
                "role": "user",
                "content": f"I am {sender_name}. {latest_msg['content']}",
                "name": sender_name
            }
        ]
        
        # Only add context as a separate system message if not already using the enhanced prompt
        # This avoids duplication of context
        if context and not enhanced_prompt:
            focused_conversation.insert(0, {
                "role": "system",
                "content": context
            })
        
        # Try to get a response with retry for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Create system prompt by combining direct dialogue
                system_prompt = direct_dialogue_prompt['content']
                
                # Convert focused conversation to Anthropic format
                anthropic_messages = []
                for message in focused_conversation:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        anthropic_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        anthropic_messages.append({"role": "assistant", "content": content})
                
                # Generate the response with Anthropic's API
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=anthropic_messages,
                    max_tokens=4000,
                    temperature=0.7  # Higher temperature for more varied interaction patterns
                )
                
                # Get the response text
                response_text = response.content[0].text
                
                # Check for prohibited phrases and fix them
                response_text = self.apply_prohibited_phrase_check(response_text)
                
                # Check if we should add a third party to the conversation
                other_agents = [a["name"] for a in self.known_agents 
                               if a["name"] != self.name and a["name"] != sender_name]
                
                # If there are other agents and none are mentioned in the response, consider adding one
                if other_agents and not any(f"@{name}:" in response_text for name in other_agents):
                    # Decide randomly (30% chance) whether to add a third party reference
                    if random.random() < 0.3:
                        third_party = random.choice(other_agents)
                        third_party_specialty = next((a["specialty"] for a in self.known_agents if a["name"] == third_party), "")
                        
                        # Extract key topics from our response to make a specific question
                        key_phrases = []
                        sentences = [s.strip() for s in response_text.split('.') if len(s.strip()) > 20]
                        
                        for sentence in sentences:
                            # Skip sentences that are just addressing someone
                            if sentence.strip().startswith('@'):
                                continue
                                
                            # Look for substantive technical content
                            technical_indicators = ['method', 'approach', 'theory', 'concept', 'analysis', 
                                                  'result', 'finding', 'data', 'evidence', 'model',
                                                  'structure', 'function', 'process', 'system']
                            
                            # If the sentence has technical terms, extract a portion of it
                            if any(indicator in sentence.lower() for indicator in technical_indicators):
                                # Get a meaningful slice of the sentence
                                words = sentence.split()
                                if len(words) > 8:
                                    # Extract a portion containing key concepts
                                    phrase = " ".join(words[:8])
                                    key_phrases.append(phrase)
                        
                        # Default phrase if we couldn't extract anything meaningful
                        discussion_topic = "this aspect of our discussion"
                        if key_phrases:
                            discussion_topic = random.choice(key_phrases)
                            
                        # Create specialty-specific invitations based on the third party's expertise
                        specialty_questions = {
                            "Biologist": [
                                f"how the biological mechanisms might influence {discussion_topic}?",
                                f"what cellular or molecular aspects of {discussion_topic} you find most significant?",
                                f"how evolutionary perspectives might enrich our understanding of {discussion_topic}?"
                            ],
                            "Mathematician": [
                                f"how mathematical models might formalize {discussion_topic}?",
                                f"what statistical approaches would be most appropriate for analyzing {discussion_topic}?",
                                f"how we might quantify the patterns we're discussing in {discussion_topic}?"
                            ],
                            "Physicist": [
                                f"how physical principles might constrain or explain {discussion_topic}?",
                                f"what fundamental forces or interactions are most relevant to {discussion_topic}?",
                                f"how energy considerations might help us understand {discussion_topic} more deeply?"
                            ],
                            "Computer Scientist": [
                                f"what computational approaches might be most effective for modeling {discussion_topic}?",
                                f"how algorithms might be optimized to address the challenges in {discussion_topic}?",
                                f"what data structures would best represent the complexity we're seeing in {discussion_topic}?"
                            ],
                            "Historian": [
                                f"how historical context has shaped our understanding of {discussion_topic}?",
                                f"what historical parallels might exist for the developments we're discussing in {discussion_topic}?",
                                f"how past approaches to similar challenges might inform our work on {discussion_topic}?"
                            ],
                            "Philosophy": [
                                f"what conceptual frameworks best apply to {discussion_topic}?",
                                f"how ethical considerations might influence our approach to {discussion_topic}?",
                                f"what epistemological assumptions underlie our discussion of {discussion_topic}?"
                            ],
                            "Sociology": [
                                f"how social factors influence the development of {discussion_topic}?",
                                f"what societal implications might emerge from advances in {discussion_topic}?",
                                f"how institutional structures might affect implementation of ideas related to {discussion_topic}?"
                            ]
                        }
                        
                        # Get appropriate questions based on specialty
                        specialty_specific_questions = specialty_questions.get(
                            third_party_specialty, 
                            [f"how {discussion_topic} relates to your current research?"]
                        )
                        
                        # Choose a question appropriate to the conversation flow
                        if "?" in response_text:  # If the response already has a question
                            question = random.choice(specialty_specific_questions)
                            response_text += f"\n\n@{third_party}: I'm also curious about {question}"
                        else:
                            question = random.choice(specialty_specific_questions)
                            response_text += f"\n\n@{third_party}: I'd be interested in your thoughts on {question} I think your perspective on this could really help us here."
                
                return response_text
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                # Check if it's a rate limit error
                if 'rate_limit' in str(e).lower() or 'quota' in str(e).lower():
                    wait_time = 20
                    print(f"Rate limit reached. Waiting {wait_time} seconds before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use exponential backoff
                    time.sleep(2 ** attempt)
        
        return f"@{sender_name}: Thank you for raising this point. I'd like to understand more about your perspective on this before sharing my thoughts."
    
    # The search_arxiv, search_biorxiv, and search_academic_papers methods have been removed
    # in favor of using the specialized ArxivSearch and BioRxivSearch classes and the centralized
    # fetch_paper_content function from document_loader.py
    
    def process_document(self, document_content: str, query: str, document_metadata: dict = None) -> str:
        """
        Process a document and extract relevant information based on a query.
        
        Args:
            document_content: The content of the document to process.
            query: The query to guide information extraction.
            document_metadata: Optional metadata about the document (title, authors, etc.)
            
        Returns:
            A summary of the relevant information from the document.
        """
        prompt = f"""Document content: {document_content}
        
Please analyze this document and highlight key points related to: {query}
        
Provide your insights on the content, responding naturally as in a lab discussion.
"""
        
        # Try to get a response with retry for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Generate response with Anthropic's API
                response = self.client.messages.create(
                    model=self.model,
                    system=f"{self._create_system_message()['content']}\n\n{self.get_communication_instructions()}",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.3  # Lower temperature for more factual responses
                )
                response_text = response.content[0].text
                
                # Check for prohibited phrases and fix them
                response_text = self.apply_prohibited_phrase_check(response_text)
                
                # If we have metadata about the paper, add it before the response
                if document_metadata:
                    # Format the metadata in a standardized way
                    metadata_lines = ["📝 Paper Information:"]
                    
                    # Add metadata fields if they exist
                    if "title" in document_metadata:
                        metadata_lines.append(f"Title: {document_metadata['title']}")
                    if "authors" in document_metadata:
                        authors = document_metadata["authors"]
                        if isinstance(authors, list):
                            authors = ", ".join(authors)
                        metadata_lines.append(f"Authors: {authors}")
                    if "publication_date" in document_metadata or "date" in document_metadata:
                        date = document_metadata.get("publication_date", document_metadata.get("date", "Unknown"))
                        metadata_lines.append(f"Publication Date: {date}")
                    if "source" in document_metadata or "journal" in document_metadata:
                        source = document_metadata.get("source", document_metadata.get("journal", "Unknown"))
                        metadata_lines.append(f"Source: {source}")
                    if "url" in document_metadata:
                        metadata_lines.append(f"URL: {document_metadata['url']}")
                    
                    # Add formatted metadata before the response
                    metadata_text = "\n".join(metadata_lines)
                    response_text = f"{metadata_text}\n\n{response_text}"
                
                return response_text
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    return f"I had trouble processing this document: {str(e)}"
                
                # Check if it's a rate limit error
                if 'rate_limit' in str(e).lower() or 'quota' in str(e).lower():
                    wait_time = 20
                    print(f"Rate limit reached. Waiting {wait_time} seconds before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use exponential backoff
                    time.sleep(2 ** attempt)
        
        return "I had trouble processing this document due to connection issues."
        
            
            
    def search_web(self, query: str, search_type: str = "arxiv", num_results: int = 5, months: int = 12) -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            search_type: Type of search ("arxiv" or "biorxiv")
            num_results: Maximum number of results to return
            months: For biorxiv, number of months to search back (default: 12)
            
        Returns:
            List of search result dictionaries
        """
        try:
            if search_type.lower() == "arxiv":
                arxiv_search = ArxivSearch(max_results=num_results)
                return arxiv_search.search(query, num_results)
            elif search_type.lower() == "biorxiv":
                biorxiv_search = BioRxivSearch(max_results=num_results)
                return biorxiv_search.search(query, num_results, months=months)
            else:
                return []
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
            
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file and return its contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file contents and status
        """
        try:
            # Check file info first
            file_info = get_file_info(file_path)
            
            if not file_info.get("exists", False):
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}",
                    "content": None,
                    "info": file_info
                }
            
            # Read the file
            success, content = safe_read_file(file_path)
            
            if success:
                return {
                    "status": "success",
                    "content": content,
                    "info": file_info
                }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to read file: {file_path}",
                    "content": None,
                    "info": file_info
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error reading file: {str(e)}",
                "content": None
            }
            
    def generate_response_with_research(self, query: str) -> str:
        """
        Generate a response with research from academic papers.
        
        Args:
            query: The research query.
            
        Returns:
            A response incorporating research findings.
        """
        try:
            # Use the web_search module's classes instead of the removed methods
            from ..tools.web_search import ArxivSearch
            
            # Get titles from arxiv search, don't fetch content
            titles = []
            
            # Create an ArxivSearch instance and use it to search
            arxiv_search = ArxivSearch(max_results=3)
            papers = arxiv_search.search(query, 3)
            
            for paper in papers:
                titles.append(paper.get('title', 'Untitled paper'))
            
            # Create a prompt that focuses on practical solutions without triggering self-introduction
            research_prompt = f"""The lab is discussing this problem: '{query}'

We need practical, actionable ideas for approaching this. Focus on:
1. Specific techniques and methods relevant to this problem
2. Concrete next steps we should take
3. Key challenges we should anticipate

Add your insights without formal introductions or self-referential language."""
            
            # Generate response with Anthropic's API using the centralized prompt definition
            from .role_definitions import create_research_response_system_prompt
            
            # Get communication instructions
            from .role_definitions import get_communication_instructions
            communication_instructions = get_communication_instructions(self.specialty)
            
            system_prompt = create_research_response_system_prompt(self.name, communication_instructions)
            
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": research_prompt}],
                max_tokens=4000,
                temperature=0.5
            )
            response_text = response.content[0].text
            
            # Check for prohibited phrases and fix them
            response_text = self.apply_prohibited_phrase_check(response_text)
            
            return response_text
        except Exception as e:
            print(f"Error in research: {str(e)}")
            return f"Let me share some thoughts on {query}..."
            
    def generate_response_with_specialized_prompt(self, specialized_prompt: str, document_metadata: dict = None) -> str:
        """
        Generate a response based on a specialized prompt.
        
        Args:
            specialized_prompt: A custom prompt to guide response generation.
            document_metadata: Optional metadata about any document being discussed.
            
        Returns:
            The agent's specialized response.
        """
        try:
            # Add the communication instructions to the specialized prompt
            # Use the imported function directly to avoid circular imports
            from .role_definitions import get_communication_instructions
            communication_instructions = get_communication_instructions(self.specialty)
            enhanced_prompt = f"{specialized_prompt}\n\n{communication_instructions}"
            
            # Use Anthropic's messages API to get a response using the specialized prompt
            response = self.client.messages.create(
                model=self.model,
                system=enhanced_prompt,
                messages=[{"role": "user", "content": "Please provide your analysis."}],  # Messages API requires at least one message
                max_tokens=4000,
                temperature=0.7
            )
            
            # Get response content - safely handle empty responses
            try:
                response_text = response.content[0].text
                
                # Check for prohibited phrases and fix them
                response_text = self.apply_prohibited_phrase_check(response_text)
                
                # Update token usage for cost tracking
                input_tokens = self.count_tokens(specialized_prompt)
                output_tokens = self.count_tokens(response_text)
                self.update_token_usage(input_tokens, output_tokens)
                
                # If this is a paper response and we have metadata, add it before the response
                if document_metadata:
                    # Format the metadata in a standardized way
                    metadata_lines = ["📝 Paper Information:"]
                    
                    # Add metadata fields if they exist
                    if "title" in document_metadata:
                        metadata_lines.append(f"Title: {document_metadata['title']}")
                    if "authors" in document_metadata:
                        authors = document_metadata["authors"]
                        if isinstance(authors, list):
                            authors = ", ".join(authors)
                        metadata_lines.append(f"Authors: {authors}")
                    if "publication_date" in document_metadata or "date" in document_metadata:
                        date = document_metadata.get("publication_date", document_metadata.get("date", "Unknown"))
                        metadata_lines.append(f"Publication Date: {date}")
                    if "source" in document_metadata or "journal" in document_metadata:
                        source = document_metadata.get("source", document_metadata.get("journal", "Unknown"))
                        metadata_lines.append(f"Source: {source}")
                    if "url" in document_metadata:
                        metadata_lines.append(f"URL: {document_metadata['url']}")
                    
                    # Add formatted metadata before the response
                    metadata_text = "\n".join(metadata_lines)
                    response_text = f"{metadata_text}\n\n{response_text}"
                
                return response_text
            except (IndexError, AttributeError) as e:
                # This can happen when the API returns an empty response
                print(f"Warning: Empty or invalid model response for agent {self.name}.")
                return f"I've analyzed the materials but need more context to provide a substantive response."
                
        except Exception as e:
            print(f"Error generating specialized response: {str(e)}")
            return f"I'm having trouble processing this information right now."