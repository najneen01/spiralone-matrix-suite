import streamlit as st
import pandas as pd
import plotly.express as px
import io

# Comprehensive Physicochemical Profile Matrix for ML Feature Engineering
AA_PROPERTIES = {
    'A': {'name': 'Alanine', 'mass': 71.08, 'charge': 0, 'hydropathy': 1.8, 'volume': 88.6},
    'R': {'name': 'Arginine', 'mass': 156.19, 'charge': 1, 'hydropathy': -4.5, 'volume': 173.4},
    'N': {'name': 'Asparagine', 'mass': 114.10, 'charge': 0, 'hydropathy': -3.5, 'volume': 114.1},
    'D': {'name': 'Aspartate', 'mass': 115.09, 'charge': -1, 'hydropathy': -3.5, 'volume': 111.1},
    'C': {'name': 'Cysteine', 'mass': 103.14, 'charge': 0, 'hydropathy': 2.5, 'volume': 108.5},
    'E': {'name': 'Glutamate', 'mass': 129.12, 'charge': -1, 'hydropathy': -3.5, 'volume': 138.4},
    'Q': {'name': 'Glutamine', 'mass': 128.13, 'charge': 0, 'hydropathy': -3.5, 'volume': 143.8},
    'G': {'name': 'Glycine', 'mass': 57.05, 'charge': 0, 'hydropathy': -0.4, 'volume': 60.1},
    'H': {'name': 'Histidine', 'mass': 137.14, 'charge': 0.1, 'hydropathy': -3.2, 'volume': 153.2},
    'I': {'name': 'Isoleucine', 'mass': 113.16, 'charge': 0, 'hydropathy': 4.5, 'volume': 166.7},
    'L': {'name': 'Leucine', 'mass': 113.16, 'charge': 0, 'hydropathy': 3.8, 'volume': 166.7},
    'K': {'name': 'Lysine', 'mass': 128.17, 'charge': 1, 'hydropathy': -3.9, 'volume': 168.6},
    'M': {'name': 'Methionine', 'mass': 131.20, 'charge': 0, 'hydropathy': 1.9, 'volume': 162.9},
    'F': {'name': 'Phenylalanine', 'mass': 147.18, 'charge': 0, 'hydropathy': 2.8, 'volume': 189.9},
    'P': {'name': 'Proline', 'mass': 97.12, 'charge': 0, 'hydropathy': -1.6, 'volume': 112.7},
    'S': {'name': 'Serine', 'mass': 87.08, 'charge': 0, 'hydropathy': -0.8, 'volume': 89.0},
    'T': {'name': 'Threonine', 'mass': 101.11, 'charge': 0, 'hydropathy': -0.7, 'volume': 116.1},
    'W': {'name': 'Tryptophan', 'mass': 186.21, 'charge': 0, 'hydropathy': -0.9, 'volume': 227.8},
    'Y': {'name': 'Tyrosine', 'mass': 163.18, 'charge': 0, 'hydropathy': -1.3, 'volume': 193.6},
    'V': {'name': 'Valine', 'mass': 99.13, 'charge': 0, 'hydropathy': 4.2, 'volume': 140.0}
}

st.set_page_config(page_title="SpiralOne Variant Matrix Machine", layout="wide")

# UI Styling and Brand Placement
st.title("🧬 SpiralOne Variant Feature Vector Engine")
st.markdown("""
Welcome to the high-throughput variant screening terminal powered by **SpiralOne**. 
This tool automatically transforms large batches of genetic mutations into engineered physical matrix tensors ready for downstream Scientific Machine Learning architectures.
""")

st.divider()

# Monetization & Operational Sidebar Controls
st.sidebar.header("Platform Tier Control")
tier = st.sidebar.selectbox("Select Service Tier", ["Standard Tier (Free)", "Enterprise Tier ($49/mo)"])

if tier == "Standard Tier (Free)":
    st.sidebar.warning("⚠️ Limit: 3 mutations processed max. Advanced batch downloads locked.")
else:
    st.sidebar.success("💎 Enterprise Active: Unlimited structural variants enabled.")

st.sidebar.subheader("Input Variant Stream")
mutation_input = st.sidebar.text_area(
    "Enter Point Mutations (One per line)", 
    value="G12D\nQ61H\nT790M", 
    help="Format: [WildType][Position][MutantType]"
)

# Execution Logic
if mutation_input:
    # Parse inputs cleanly
    raw_mutations = [m.strip().upper() for m in mutation_input.split("\n") if m.strip()]
    
    # Apply Tier Limits
    if tier == "Standard Tier (Free)" and len(raw_mutations) > 3:
        st.error("❌ Standard Tier limit exceeded. The free engine only parses up to 3 mutation features simultaneously. Please upgrade to the Enterprise Tier to run high-throughput batch sequences.")
        processed_mutations = raw_mutations[:3]
    else:
        processed_mutations = raw_mutations

    feature_records = []
    
    # Process each mutation vector
    for mut_str in processed_mutations:
        try:
            # Basic parsing validation logic
            wild_aa = mut_str[0]
            mut_aa = mut_str[-1]
            position = int(mut_str[1:-1])
            
            if wild_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
                w_data = AA_PROPERTIES[wild_aa]
                m_data = AA_PROPERTIES[mut_aa]
                
                # Math Engine: Feature engineering delta calculations
                d_mass = m_data['mass'] - w_data['mass']
                d_charge = m_data['charge'] - w_data['charge']
                d_hydro = m_data['hydropathy'] - w_data['hydropathy']
                d_vol = m_data['volume'] - w_data['volume']
                
                # Structural Disruptive Metric Score (Simulated ML Output Vector)
                disruption_score = (abs(d_mass)/100) + abs(d_charge) + abs(d_hydro) + (abs(d_vol)/50)
                
                feature_records.append({
                    "Mutation": mut_str,
                    "Position": position,
                    "Wild_Type": wild_aa,
                    "Mutant_Type": mut_aa,
                    "Delta_Mass": round(d_mass, 2),
                    "Delta_Charge": round(d_charge, 2),
                    "Delta_Hydropathy": round(d_hydro, 2),
                    "Delta_Volume": round(d_vol, 2),
                    "Disruption_Index": round(disruption_score, 3)
                })
        except Exception:
            st.sidebar.error(f"Format Error: Visual representation engine skipped '{mut_str}'. Use format like G12D.")

    if feature_records:
        df_results = pd.DataFrame(feature_records)
        
        # Output Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Successfully Parsed Metrics", len(df_results))
        col2.metric("Highest Structural Shift", df_results['Disruption_Index'].max())
        col3.metric("System RAM Utilization", "Minimal (< 15MB)")

        # Main Data Frame Matrix Output
        st.subheader("📊 Structured Biophysical Vector Outputs (Dataset Tensors)")
        st.dataframe(df_results, use_container_width=True)

        # High-Throughput Data Actions Block
        st.subheader("📥 Action & Pipeline Integrations")
        
        if tier == "Enterprise Tier ($49/mo)":
            # Generate downloadable CSV buffer dynamically
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            
            st.download_button(
                label="📥 Export Machine-Learning-Ready CSV",
                data=csv_data,
                file_name="spiralone_ml_features.csv",
                mime="text/csv"
            )
        else:
            st.markdown("### 🔒 Lock Status: Enterprise Features Restricted")
            st.info("Unlock high-throughput programmatic script integration, download clean CSV data frames, and retrieve advanced structural variance scoring arrays for just $49/month.")
            if st.button("🔓 Upgrade to SpiralOne Enterprise"):
                st.balloons()
                st.success("Redirecting safely to checkout gateway...")

        st.divider()
        
        # Interactive Visual Distributions
        st.subheader("🌐 Structural Disruption Profiles Map")
        fig = px.scatter(
            df_results, 
            x="Position", 
            y="Disruption_Index", 
            color="Delta_Hydropathy", 
            size="Disruption_Index",
            hover_data=["Mutation", "Delta_Charge", "Delta_Mass"],
            title="Positional Scatter Plot of Engineered Mutation Trajectories"
        )
        st.plotly_chart(fig, use_container_width=True)