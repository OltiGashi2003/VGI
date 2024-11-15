import streamlit as st

def run():
    # About Us Page Header
    st.title("About Us - Pathfinder")
    st.subheader("Get there, no matter where.")

    # Vision Section
    st.header("Our Vision")
    st.write("""
        Pathfinder app was developed with a clear purpose: to enhance the VGI-Flexi transportation model for the Ingolstadt Regional Area, 
        making on-demand transit more efficient and accessible.
    """)

    # Services Section
    st.header("Our Services")
    st.write("""
       With VGI-provided data, we've visualized essential insights, including high-demand zones, hotspots, and the patterns behind canceled and completed trips. 
       Through analyzing these data points and recognizing the high cancellation rates, we set out to create not just a visualization tool, but a fully integrated solution featuring three core innovations:
    """)

    # Key Features
    st.subheader("Key Features")
    
    # Feature 1: Demand Prediction
    st.markdown("**Demand Prediction**")
    st.write("""
        “With the Prophet model, we forecast demand trends, empowering operators to anticipate needs for the next day, the following week, and peak periods. 
        This predictive capability helps reduce service gaps and improve scheduling.”
    """)

    # Feature 2: Strategic Bus Repositioning
    st.markdown("**Strategic Bus Repositioning**")
    st.write("""
       “Because Flexi buses lack static positioning, our algorithm strategically places buses closer to anticipated pickup areas. 
       This readiness reduces wait times and minimizes cancellations, aligning resources where they’re needed most.”
    """)

    # Feature 3: Optimized Routing for Speed and Sustainability
    st.markdown("**Optimized Routing**")
    st.write("""
        “Pathfinder calculates routes that are both fast and carbon-efficient, supporting timely pickups while reducing emissions—a crucial aspect for sustainable rural mobility.”
    """)

    # Team Section
    st.header("Our Team")
    st.write("""“Pathfinder calculates routes that prioritize both speed and carbon efficiency, supporting timely pickups while reducing emissions. 
    This feature is essential for sustainable rural mobility, offering an eco-friendly solution to on-demand transit.”
    """)

    # Create three columns for the team members
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("olti.jpeg", width=150)  # Replace with the path to the image of the first team member
        st.markdown("Olti Gashi")
        st.write("THI Student, Data Science, 5th semester")

    with col2:
        st.image("driti.jpeg", width=150)  # Replace with the path to the image of the second team member
        st.markdown("Driti Sanaja")
        st.write("THI Student, Computer Science & Artificial Intelligence, 5th semester")

    with col3:
        st.image("shendi.jpeg", width=150)  # Replace with the path to the image of the third team member
        st.markdown("Shend Sanaja")
        st.write("THI Student, Computer Science & Artificial Intelligence, 7th semester")