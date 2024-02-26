import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pytest
from lib.agent import Agent, communicate

def test_agent_initialization():
    # Test the initialization of Agent Alice
    alice = Agent(id=0, name="Alice", age=21, hobby="Reading", job="Student")
    assert alice.name == "Alice"
    assert alice.age == 21
    assert alice.hobby == "Reading"
    assert alice.job == "Student"
    
    # Test the initialization of Agent Bob
    bob = Agent(id=1, name="Bob", age=35, hobby="Hiking", job="Teacher")
    assert bob.name == "Bob"
    assert bob.age == 35
    assert bob.hobby == "Hiking"
    assert bob.job == "Teacher"

def test_agent_interview():
    # Create agents
    alice = Agent(id=0, name="Alice", age=21, hobby="Reading", job="Student")
    bob = Agent(id=1, name="Bob", age=35, hobby="Hiking", job="Teacher")
    
    # Mock interview responses
    alice_response = alice.interview('What have you been up to recently?')
    bob_response = bob.interview('What have you been up to recently?')
    
    # Test the interview responses
    assert isinstance(alice_response, str)
    assert isinstance(bob_response, str)

def test_memory_addition():
    alice = Agent(id=0, name="Alice", age=21, hobby="Reading", job="Student")
    bob = Agent(id=1, name="Bob", age=35, hobby="Hiking", job="Teacher")
    
    # Test adding memory to Alice
    alice.memory.add_memory(memory_content="I am struggling with the homework.")
    assert "I am struggling with the homework." in [memory.page_content for memory in alice.memory.memory_stream]  # Assumes that Agent has a 'memories' list
    
    # Test adding memory to Bob
    bob.memory.add_memory(memory_content="I am great at giving advice.")
    assert "I am great at giving advice." in [memory.page_content for memory in bob.memory.memory_stream]

def test_communication():
    alice = Agent(id=0, name="Alice", age=21, hobby="Reading", job="Student")
    bob = Agent(id=1, name="Bob", age=35, hobby="Hiking", job="Teacher")
    
    # Test communication between Alice and Bob
    conversation_result = communicate(caller=alice, callee=bob, question="What is the capital of France?")
    
    # Test the result of the communication
    assert conversation_result is not None
