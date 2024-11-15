import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def run():
    st.title("Pathfinder App")
    
    # Create a layout with two columns for the two products
    col1, col2 = st.columns(2)

    with col1:
        # Add image and description for Forecasting Demand
        st.subheader("📈 Demand Forecasting")
        st.image("forecasting_demand_image.jpg", use_container_width =True)  # Replace with the path to your image
        st.write("""
        This product predicts the demand for public transportation pickups in the VGI region.
        Using machine learning, it forecasts future demand based on historical data.
        Click on the image to start using the product.
        """)
        # Add clickable image to redirect to Products page
        # if st.button("Go to Forecasting Demand"):
        #     st.session_state.page = "Products"
        #     st.session_state.selected_product = "Forecasting Demand"
        #     switch_page("Products")


    with col2:
        # Add image and description for Actual Demand
        st.subheader("📊 Actual Demand")
        st.image("vgi_flexi_dashboard.png", use_container_width=True)
        # st.image("actual_demand_image.jpg", use_column_width=True)  # Replace with the path to your image
        st.write("""
        This product helps visualize actual demand for pickups in the VGI region.
        It shows the real-time or historical demand data, allowing you to analyze patterns.
        Click on the image to start using the product.
        """)
        # Add clickable image to redirect to Products page
        # if st.button("Go to Actual Demand"):
        #     st.session_state.page = "Products"
        #     st.session_state.selected_product = "Actual Demand"
        #     switch_page("Products")
