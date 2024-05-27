import pytest
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import Agent, communicate

def test_agent_initialization():
    # Test the initialization of Agent Alice
    alice = Agent(id=0, name="Alice")
    assert alice.name == "Alice"
    
    # Test the initialization of Agent Bob
    bob = Agent(id=1, name="Bob")
    assert bob.name == "Bob"

def test_agent_interview():
    # Create agents
    alice = Agent(id=0, name="Alice")
    bob = Agent(id=1, name="Bob")
    
    # Mock interview responses
    alice_response = alice.interview('What have you been up to recently?')
    bob_response = bob.interview('What have you been up to recently?')
    
    # Test the interview responses
    assert isinstance(alice_response, str)
    assert isinstance(bob_response, str)

def test_memory_addition():
    alice = Agent(id=0, name="Alice")
    bob = Agent(id=1, name="Bob")
    
    # Test adding memory to Alice
    alice.memory.add_memory(memory_content="I am struggling with the homework.")
    assert "I am struggling with the homework." in [memory.page_content for memory in alice.memory.memory_stream]  # Assumes that Agent has a 'memories' list
    
    # Test adding memory to Bob
    bob.memory.add_memory(memory_content="I am great at giving advice.")
    assert "I am great at giving advice." in [memory.page_content for memory in bob.memory.memory_stream]

def test_communication():
    alice = Agent(id=0, name="Alice")
    bob = Agent(id=1, name="Bob")
    
    # Test communication between Alice and Bob
    conversation_result = communicate(caller=alice, callee=bob, question="What is the capital of France?")
    
    # Test the result of the communication
    assert conversation_result is not None
