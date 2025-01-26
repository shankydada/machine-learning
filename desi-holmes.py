# Function to generate detailed observations, leads, and actions
def generate_case_analysis(new_case_description):
    # Transform the new case description
    new_case_vector = vectorizer.transform([new_case_description])
    
    # Compute similarity with all previous cases
    case_vectors = vectorizer.transform(df['description'])
    similarities = cosine_similarity(new_case_vector, case_vectors).flatten()
    
    # Get top matching cases
    top_indices = similarities.argsort()[-5:][::-1]  # Top 5 most similar cases
    similar_cases = df.iloc[top_indices]
    
    # Generate unique observations
    observations = []
    leads = []
    for i, case in similar_cases.iterrows():
        # Identify specific keywords or patterns shared between the cases
        common_keywords = set(new_case_description.split()).intersection(set(case['description'].split()))
        common_keywords_str = ', '.join(common_keywords) if common_keywords else "No significant keywords found"
        
        observations.append(f"Case '{case['description']}' has status '{case['status']}' with common keywords: {common_keywords_str}.")
        
        leads.append({
            "Lead": f"Derived from case '{case['description']}'",
            "Confidence Score": round(similarities[i] * 100, 2),  # Use `i` directly
            "Suggested Action": f"Investigate patterns or evidence similar to case '{case['description']}'."
        })
    
    # Default next steps
    next_steps = [
        "Compare forensic evidence with similar past cases.",
        "Conduct additional witness interviews to gather insights.",
        "Leverage AI tools for deeper pattern analysis in unresolved leads."
    ]
    
    return {
        "Observations": observations,
        "Leads": leads,
        "Next Steps": next_steps
    }

# Input the case description
new_case_description = input("\nPlease enter the case description: ")

# Predict the solution
predicted_status = predict_case_solution(new_case_description)
print(f"\nThe predicted status for this case is: {predicted_status}")

# Generate detailed case analysis
analysis = generate_case_analysis(new_case_description)

# Output the detailed observations, leads, and next steps
print("\nObservations:")
for observation in analysis["Observations"]:
    print(f"- {observation}")

print("\nLeads:")
for lead in analysis["Leads"]:
    print(f"Lead: {lead['Lead']}")
    print(f"Confidence Score: {lead['Confidence Score']}")
    print(f"Suggested Action: {lead['Suggested Action']}")
    print()

print("Next Steps:")
for step in analysis["Next Steps"]:
    print(f"- {step}")

