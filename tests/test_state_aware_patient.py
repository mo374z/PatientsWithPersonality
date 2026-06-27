"""
Simple test script for StateAwarePatient class
"""

import asyncio
from patient_simulator.patients import StateAwarePatient


async def main():
    """Test the StateAwarePatient with a simple case"""

    case_description = """
    A 45-year-old male patient presents with chest pain that started 2 hours ago.
    The pain is described as pressure-like, radiating to the left arm.
    Patient has a history of hypertension and high cholesterol.
    Patient is a smoker (1 pack per day for 20 years).
    Blood pressure: 150/95 mmHg
    Heart rate: 95 bpm
    """

    # Create patient simulator
    patient = StateAwarePatient(
        case_description=case_description.strip(), model="gemini-2.5-flash"
    )

    print("=" * 60)
    print("Testing State Aware Patient Simulator")
    print("=" * 60)
    print()

    # Test 1: Initialization (first turn)
    print("Test 1: Initialization")
    print("-" * 60)
    doctor_q1 = "Hello, I'm your doctor. How can I help you today?"
    print(f"Doctor: {doctor_q1}")
    response1 = await patient.get_response(doctor_q1)
    print(f"Patient: {response1}")
    print()

    # Test 2: Effective inquiry
    print("Test 2: Effective Inquiry (specific question with relevant info)")
    print("-" * 60)
    doctor_q2 = "Can you describe the chest pain in more detail?"
    print(f"Doctor: {doctor_q2}")
    response2 = await patient.get_response(doctor_q2)
    print(f"Patient: {response2}")
    print()

    # Test 3: Ineffective inquiry
    print("Test 3: Ineffective Inquiry (specific but no relevant info)")
    print("-" * 60)
    doctor_q3 = "Do you have any allergies to medications?"
    print(f"Doctor: {doctor_q3}")
    response3 = await patient.get_response(doctor_q3)
    print(f"Patient: {response3}")
    print()

    # Test 4: Ambiguous inquiry
    print("Test 4: Ambiguous Inquiry (too broad)")
    print("-" * 60)
    doctor_q4 = "Tell me everything about your health."
    print(f"Doctor: {doctor_q4}")
    response4 = await patient.get_response(doctor_q4)
    print(f"Patient: {response4}")
    print()

    # Test 5: Effective advice
    print("Test 5: Effective Advice (with relevant results)")
    print("-" * 60)
    doctor_q5 = "I recommend checking your blood pressure."
    print(f"Doctor: {doctor_q5}")
    response5 = await patient.get_response(doctor_q5)
    print(f"Patient: {response5}")
    print()

    # Test 6: Demand (physical action)
    print("Test 6: Demand (physical action)")
    print("-" * 60)
    doctor_q6 = "Can you please open your mouth so I can examine your throat?"
    print(f"Doctor: {doctor_q6}")
    response6 = await patient.get_response(doctor_q6)
    print(f"Patient: {response6}")
    print()

    # Test 7: Other topics
    print("Test 7: Other Topics (off-topic question)")
    print("-" * 60)
    doctor_q7 = "What's your favorite movie?"
    print(f"Doctor: {doctor_q7}")
    response7 = await patient.get_response(doctor_q7)
    print(f"Patient: {response7}")
    print()

    # Test 8: Conclusion
    print("Test 8: Conclusion")
    print("-" * 60)
    doctor_q8 = (
        "Thank you for your time. Take care and follow the treatment plan. Goodbye."
    )
    print(f"Doctor: {doctor_q8}")
    response8 = await patient.get_response(doctor_q8)
    print(
        f"Patient: {response8 if response8 else '(No response - consultation ended)'}"
    )
    print()

    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
