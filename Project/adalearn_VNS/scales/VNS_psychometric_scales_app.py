"To execute the code, navigate to correct directory level then in the terminal >>> streamlit run VNS_psychometic_scales_app.py"

import streamlit as st
import pandas as pd

# Setting structure and navigation 
st.set_page_config(page_title="Questionnaires")

# Initialize session state for participant info if needed
if 'info_collected' not in st.session_state:
    st.session_state.info_collected = False

# Conditional navigation based on time point
if "page" not in st.session_state:
    st.session_state.page = "Informations du participant"

nav_pages = ["Informations du participant", "EVA", "Téléchargement"]

st.session_state.page = st.sidebar.radio(
    "Navigation",
    nav_pages,
    index=nav_pages.index(st.session_state.page) if st.session_state.page in nav_pages else 0
)

page = st.session_state.page

# --- Page 1: Participant Information ---
if page == "Informations du participant":
    st.title("Informations du participant")

    time_points = {
        'time1': 'Baseline (H:0)',
        'time2': 'Post task (H:3)'    
        }

    with st.form("participant_info_form"):
        participant_id = st.text_input("ID du participant")
        session_code = st.text_input("Code de session")

        submit_info = st.form_submit_button("Commencer le questionnaire")

        if submit_info:
            if participant_id.strip() == "" or session_code.strip() == "":
                st.warning("Merci de remplir tous les champs.")
            else:
                st.session_state.participant_id = participant_id
                st.session_state.session_code = session_code
                st.session_state.info_collected = True
                st.session_state.page = "EVA"
                st.stop()

# --- Page 2: Visual Analogue Scales ---
elif page == "EVA":
    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
    else:
        if "eva_results_df" not in st.session_state:
            st.session_state.eva_results_df = None

        st.markdown("""<div style='font-size: 32px; font-weight: bold;'>EVA</div>""", unsafe_allow_html=True)

        with st.form("eva_form"):

            st.markdown("""<div style='font-size: 16px;'> 
            <b>CONSIGNES</b>: Veuillez indiquer sur la ligne horizontale dans quelle mesure vous étiez conscient de la stimulation et du niveau de douleur ressenti à cause de la stimulation pendant l'expérience.</b>. <br><br><br>
            </div>""", unsafe_allow_html=True)

            conscience = st.slider("Conscience de la stimulation", min_value=0, max_value=100, value=50, step=1)
            douleur = st.slider("Douleur de stimulation", min_value=0, max_value=100, value=50, step=1)

            submit_vas = st.form_submit_button("Soumettre")

            if submit_vas:
                vas_data = {
                    "Conscience": conscience,
                    "Douleur": douleur,
                }

                vas_df = pd.DataFrame([vas_data])

                vas_df.insert(0, 'participant_id', st.session_state.participant_id)
                vas_df.insert(1, 'session_code', st.session_state.session_code)

                # Save to session state to keep data persistent across reruns
                st.session_state.eva_results_df = vas_df

        if st.session_state.eva_results_df is not None:
            if st.button("Voir les résultats EVA"):
                st.subheader("Réponses EVA")
                st.dataframe(st.session_state.eva_results_df)
                st.success(f"Score Conscience: {st.session_state.eva_results_df['Conscience'].iloc[0]}")
                st.success(f"Score Douleur: {st.session_state.eva_results_df['Douleur'].iloc[0]}")

            if st.button("Suivant"):
                st.session_state.page = "Téléchargement"  # your next page key
                #st.session_state._rerun_triggered = True
                st.stop()

# --- Page 3: Download data ---
elif page == "Téléchargement":
    st.markdown("""<div style='font-size: 32px; font-weight: bold;'>Téléchargement</div>""", unsafe_allow_html=True)

    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
        st.stop()      

    required_keys = ["eva_results_df"]

    all_completed = all(st.session_state.get(k) is not None for k in required_keys)
        
    dataframes = {
            "eva_results_df": "EVA",
        }

    for key, label in dataframes.items():
        df = st.session_state.get(key)

        if df is not None:
            st.subheader(f"{label} Résultats")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Télécharger {label}.csv",
                data=csv_data,
                file_name=f"{st.session_state.participant_id}_{st.session_state.session_code}_{label}_.csv",
                mime='text/csv'
            )
        else:
            # Only show missing info if it was expected
            if key in required_keys:
                st.info(f"Aucun résultat disponible pour {label}.")
