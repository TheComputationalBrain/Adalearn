"To execute the code, navigate to correct directory level then in the terminal >>> streamlit run psychometic_scales_app.py"

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

# Define which pages should be available based on time point
if "assessment_time" in st.session_state:
    if st.session_state.assessment_time == "Baseline (H:0)":
        nav_pages = ["Informations du participant", "STAI-Y (État)", "STAI-Y (Trait)", "BDI-II", "EVA", "Téléchargement"]
    elif st.session_state.assessment_time == "Post task (H:3)":
        nav_pages = ["Informations du participant", "STAI-Y (État)", "EVA", "Téléchargement"]
    else:
        nav_pages = ["Informations du participant"]
else:
    nav_pages = ["Informations du participant"]

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

        # Use selectbox to choose from the time_points
        selected_label = st.selectbox(
            "Moment de l'évaluation",
            options=list(time_points.values())
        )

        submit_info = st.form_submit_button("Commencer le questionnaire")

        if submit_info:
            if participant_id.strip() == "" or session_code.strip() == "":
                st.warning("Merci de remplir tous les champs.")
            else:
                st.session_state.participant_id = participant_id
                st.session_state.session_code = session_code
                st.session_state.assessment_time = selected_label  
                st.session_state.info_collected = True
                st.session_state.page = "STAI-Y (État)"
                #st.session_state._rerun_triggered = True
                st.stop()


# --- Page 2: STAI-Y (State) Questionnaire ---
elif page == "STAI-Y (État)":
    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
    else:
        st.markdown("<div style='font-size: 32px; font-weight: bold;'>STAI-Y (État)</div>", unsafe_allow_html=True)

        if "stai_results_df" not in st.session_state:
            st.session_state.stai_results_df = None

        # Define items
        items = {
            f"item{i+1}": question for i, question in enumerate([
                # First 20 – State anxiety
                "Je me sens calme.",
                "Je me sens en sécurité.",
                "Je suis tendu(e), crispé(e).",
                "Je me sens surmené(e).",
                "Je me sens tranquille, bien dans ma peau.",
                "Je me sens ému(e), bouleversé(e).",
                "L'idée de malheurs éventuels me tracasse en ce moment",
                "Je me sens content(e).",
                "Je me sens effrayé(e).",
                "Je me sens à mon aise.",
                "Je me sens que j'ai confiance en moi.",
                "Je me sens nerveux(nerveuse), irritable.",
                "J'ai la frousse, la trouille (j'ai peur).",
                "Je me sens indécis(e).",
                "Je suis décontracté(e), détendu(e).",
                "Je me sens satisfait(e).",
                "Je suis inquiet, soucieux (inquiéte, soucieuse).",
                "Je ne sais plus où j'en suis, je me sens décontracté(e), dérouté(e).",
                "Je me sens solide, posé(e), pondéré(e), réfléchi(e).",
                "Je me sens dans de bonnes humeur, aimable.",
            ])
        }

        reverse_items = ['item1', 'item2', 'item5', 'item8', 'item10', 'item11', 'item15', 'item16', 'item19', 'item20']

        likert_options = {
            "Non": 1,
            "Plutôt non": 2,
            "Plutôt oui": 3,
            "Oui": 4
        }

        responses = {f"item{i}": None for i in range(1, 21)}  # initialize all

        with st.form("state_form"):
            st.markdown("""<div style='font-size: 16px;'>
                         <b>CONSIGNES</b>: Un certain nombre de phrases que l'on utilise pour se décrire sont données ci-dessous. Lisez chaque phrase, puis marquez d'une croix, parmi les quatre points à droite, celui qui correspond le mieux à ce que vous ressentez <b>à l'instant, juste en ce moment</b>. Il n'y a pas de bonnes ni de mauvaises réponses. Ne passez pas trop de temps sur l'une ou l'autre de ces propositions, et indiquez la réponse qui décrit le mieux vos sentiments <b>actuels</b>. <br><br><br>
                        </div>""", unsafe_allow_html=True)

            # Show all state items
            for i in range(1, 21):
                key = f"item{i}"
                resp = st.radio(items[key], list(likert_options.keys()), index=None, key=key, horizontal=True)
                responses[key] = likert_options[resp] if resp is not None else None

            submit_state = st.form_submit_button("Soumettre")

            if submit_state:
                if any(responses[f"item{i}"] is None for i in range(1, 21)):
                    st.warning("Veuillez répondre à toutes les questions avant de soumettre.")
                else:
                    # Create DataFrame for state responses
                    stai_df = pd.DataFrame([responses])

                    # Reverse-score items
                    for item in reverse_items:
                        if item in stai_df:
                            stai_df[item] = 5 - stai_df[item]

                    # Compute state score
                    stai_df['score_state'] = stai_df[[f'item{i}' for i in range(1, 21)]].sum(axis=1)

                    # Add participant info
                    stai_df.insert(0, 'participant_id', st.session_state.participant_id)
                    stai_df.insert(1, 'session_code', st.session_state.session_code)
                    stai_df.insert(2, 'assessment_time', st.session_state.assessment_time)

                    # Save to session state
                    st.session_state.stai_results_df = stai_df
                    st.success("Questionnaire État soumis avec succès !")

                    # Move to next page automatically
                    if st.session_state.assessment_time == "Baseline (H:0)":
                        st.session_state.page = "STAI-Y (Trait)"
                    else:
                        st.session_state.page = "EVA"
                    st.stop()


# --- Page 3: STAI-Y (Trait) Questionnaire ---
elif page == "STAI-Y (Trait)":
    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
    elif st.session_state.assessment_time != "Baseline (H:0)":
        st.info("Le questionnaire Trait n'est disponible qu'au Baseline (H:0).")
    else:
        st.markdown("<div style='font-size: 32px; font-weight: bold;'>STAI-Y (Trait)</div>", unsafe_allow_html=True)

        if "stai_results_df" not in st.session_state or st.session_state.stai_results_df is None:
            st.warning("Veuillez d'abord compléter le questionnaire État.")
        else:
            # Define items
            items = {
                f"item{i+21}": question for i, question in enumerate([
                    # Next 20 – Trait anxiety
                    "Je me sens dans de bonnes humeur, aimable.",
                    "Je me sens nerveux(nerveuse), irritable et agité(e).",
                    "Je me sens content(e) de moi.",
                    "Je voudrais être aussi heureux (heureuse) que les autres semblent l'être.",
                    "J'ai un sentiment d'échec.",
                    "Je me sens reposé(e).",
                    "J'ai tout mon sang-froid.",
                    "J'ai l'impression que les difficultés s'accumulent à un tel point que je ne peux plus les surmonter.",
                    "Je m'inquiète à propos de choses sans importance.",
                    "Je suis heureux(se).",
                    "J'ai des pensées qui me perturbent.",
                    "Je manque de confiance en moi.",
                    "Je me sens sans inquiétude, en sécurité, en sûreté.",
                    "Je prends facilement des décisions.",
                    "Je me sens incompétent(e), pas à la hauteur.",
                    "Je suis satisfait(e).",
                    "Des idées sans importance trottant dans ma tête me dérangent.",
                    "Je prends les déceptions tellement à cœur que je les oublie difficilement.",
                    "Je suis une personne posée, solide, stable.",
                    "Je deviens tendu(e) et agité(e) quand je réfléchis à mes soucis."
                ])
            }

            reverse_items = ['item21', 'item23', 'item26', 'item27', 'item30', 'item33', 'item34', 'item36', 'item39']    

            likert_options = {
                    "Non": 1,
                    "Plutôt non": 2,
                    "Plutôt oui": 3,
                    "Oui": 4
                }

            responses = {key: None for key in items.keys()}

            # Form for trait questionnaire
            with st.form("trait_form"):
                st.markdown("""<div style='font-size: 16px;'>
                             <b>CONSIGNES</b>: Un certain nombre de phrases que l'on utilise pour se décrire sont données ci-dessous. Lisez chaque phrase, puis marquez d'une croix, parmi les quatre points à droite, celui qui correspond le mieux à ce que vous ressentez <b>généralement</b>. Il n'y a pas de bonnes ni de mauvaises réponses. Ne passez pas trop de temps sur l'une ou l'autre de ces propositions, et indiquez la réponse qui décrit le mieux vos sentiments <b>habituels</b>.<br><br><br>
                            </div>""", unsafe_allow_html=True)

                # Show all trait items
                for key, question in items.items():
                    resp = st.radio(question, list(likert_options.keys()), index=None, key=key, horizontal=True)
                    responses[key] = likert_options[resp] if resp is not None else None

                # Submit button inside the form
                submit_trait = st.form_submit_button("Soumettre")

                if submit_trait:
                    if any(v is None for v in responses.values()):
                        st.warning("Veuillez répondre à toutes les questions avant de soumettre.")
                    else:
                        # Copy existing state scores
                        stai_df = st.session_state.stai_results_df.copy()

                        # Add trait responses
                        for key, val in responses.items():
                            stai_df[key] = val

                        # Apply reverse scoring
                        for item in reverse_items:
                            if item in stai_df:
                                stai_df[item] = 5 - stai_df[item]

                        # Compute Trait and total scores
                        stai_df['score_trait'] = stai_df[[f'item{i}' for i in range(21, 41)]].sum(axis=1)
                        stai_df['score_total'] = stai_df['score_state'] + stai_df['score_trait']

                        # Update session state
                        st.session_state.stai_results_df = stai_df
                        st.success("Questionnaire Trait soumis avec succès !")

                        # Move to next page immediately
                        st.session_state.page = "BDI-II"
                        st.stop()      

# --- Page 4: BDI Questionnaire ---
elif page == "BDI-II":
    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
    else:
        # Only show BDI-II if Baseline
        if st.session_state.assessment_time == "Baseline (H:0)":
            
            if "bdi_results_df" not in st.session_state:
                st.session_state.bdi_results_df = None

            st.markdown("""<div style='font-size: 32px; font-weight: bold;'>BDI-II</div>""", unsafe_allow_html=True)

            bdi_ii_fr = {
                    "1. Tristesse": {
                        0: "Je ne me sens pas triste",
                        1: "Je me sens très souvent triste",
                        2: "Je suis tout le temps triste",
                        3: "Je suis si triste ou malheureux(se), que ce ne n'est pas supportable"
                    },
                    "2. Pessimisme": {
                        0: "Je ne suis pas découragé(e) face à mon avenir",
                        1: "Je me sens plus découragé(e) qu'avant face à mon avenir",
                        2: "Je ne m'attends pas à ce que les choses s'arrangent pour moi",
                        3: "J'ai le sentiment que mon avenir est sans espoir et qu'il ne peut qu'empirer"
                    },
                    "3. Échec dans le passé": {
                        0: "Je n'ai pas le sentiment d'avoir échoué dans la vie, d'être un(e) raté(e)",
                        1: "J'ai échoué plus souvent que je n'aurais dû",
                        2: "Quand je pense à mon passé, je constate un grand nombre d'échecs",
                        3: "J'ai le sentiment d'avoir complètement raté ma vie"
                    },
                    "4. Perte de plaisir": {
                        0: "J'éprouve toujours autant de plaisir qu'avant aux choses qui me plaisent",
                        1: "Je n'éprouve pas autant de plaisir aux choses qu'avant",
                        2: "J'éprouve très peu de plaisir aux choses qui me plaisaient habituellement",
                        3: "Je n'éprouve aucun plaisir aux choses qui me plaisaient habituellement"
                    },
                    "5. Sentiment de culpabilité": {
                        0: "Je ne me sens pas particulièrement coupable",
                        1: "Je me sens coupable pour bien des choses que j'ai faites ou que j'aurais dû faire",
                        2: "Je me sens assez coupable la plupart du temps",
                        3: "Je me sens tout le temps coupable"
                    },
                    "6. Sentiment d'être puni(e)": {
                        0: "Je n'ai pas le sentiment d'être puni(e)",
                        1: "Je sens que je pourrais être puni(e)",
                        2: "Je m'attends à être puni(e)",
                        3: "J'ai le sentiment d'être puni(e)"
                    },
                    "7. Sentiments négatifs envers soi-même": {
                        0: "Mes sentiments envers moi-même n'ont pas changé",
                        1: "J'ai perdu confiance en moi",
                        2: "Je suis déçu(e) par moi-même",
                        3: "Je ne m'aime pas du tout"
                    },
                    "8. Attitude critique envers soi": {
                        0: "Je ne me blâme pas ou ne me critique pas plus que d'habitude",
                        1: "Je suis plus critique envers moi-même que je ne l'étais",
                        2: "Je me reproche tous mes défauts",
                        3: "Je me reproche tous les malheurs qui arrivent"
                    },
                    "9. Pensées ou désirs de suicide": {
                        0: "Je ne pense pas du tout à me suicider",
                        1: "Il m'arrive de penser à me suicider, mais je ne le ferais pas",
                        2: "J'aimerais me suicider",
                        3: "Je me suiciderais si l'occasion se présentait"
                    },
                    "10. Pleurs": {
                        0: "Je ne pleure pas plus qu'avant",
                        1: "Je pleure plus qu'avant",
                        2: "Je pleure pour la moindre petite chose",
                        3: "Je voudrais pleurer mais je n'en suis pas capable"
                    },
                    "11. Agitation": {
                        0: "Je ne suis pas plus agité(e) ou plus tendu(e) que d'habitude",
                        1: "Je me sens plus agité(e) ou plus tendu(e) que d'habitude",
                        2: "Je suis si agité(e) ou tendu(e) que j'ai du mal à rester tranquille",
                        3: "Je suis si agité(e) ou tendu(e) que je dois continuellement bouger ou faire quelque chose"
                    },
                    "12. Perte d'intérêt": {
                        0: "Je n'ai pas perdu d'intérêt pour les gens ou pour les activités",
                        1: "Je m'intéresse moins qu'avant aux gens et aux choses",
                        2: "Je ne m'intéresse presque plus aux gens et aux choses",
                        3: "J'ai du mal à m'intéresser à quoi que ce soit"
                    },
                    "13. Indécision": {
                        0: "Je prends des décisions toujours aussi bien qu'avant",
                        1: "Il m'est plus difficile que d'habitude de prendre des décisions",
                        2: "J'ai beaucoup plus de mal qu'avant à prendre des décisions",
                        3: "J'ai du mal à prendre n'importe quelle décision"
                    },
                    "14. Dévalorisation": {
                        0: "Je pense être quelqu'un de valable",
                        1: "Je ne crois pas avoir autant de valeur ni être aussi utile qu'avant",
                        2: "Je me sens moins valable que les autres",
                        3: "Je sens que je ne vaux absolument rien"
                    },
                    "15. Perte d'énergie": {
                        0: "J'ai toujours autant d'énergie qu'avant",
                        1: "J'ai moins d'énergie qu'avant",
                        2: "Je n'ai pas assez d'énergie pour pouvoir faire grand-chose",
                        3: "Je n'ai pas assez d'énergie pour faire quoi que ce soit"
                    },
                    "16. Modifications dans les habitudes de sommeil": {
                        0: "Mes habitudes de sommeil n'ont pas changé",
                        1: "Je dors un peu plus/moins que d'habitude",
                        2: "Je dors beaucoup plus/moins que d'habitude",
                        3: "Je dors presque toute la journée / Je me réveille une ou deux heures plus tôt et je suis incapable de me rendormir"
                    },
                    "17. Irritabilité": {
                        0: "Je ne suis pas plus irritable que d'habitude",
                        1: "Je suis plus irritable que d'habitude",
                        2: "Je suis beaucoup plus irritable que d'habitude",
                        3: "Je suis constamment irritable"
                    },
                    "18. Modification de l'appétit": {
                        0: "Mon appétit n'a pas changé",
                        1: "J'ai un peu plus/moins d'appétit que d'habitude",
                        2: "J'ai beaucoup plus/moins d'appétit que d'habitude",
                        3: "Je n'ai pas d'appétit du tout / J'ai constamment envie de manger"
                    },
                    "19. Difficulté à se concentrer": {
                        0: "Je parviens à me concentrer toujours aussi bien qu'avant",
                        1: "Je ne parviens pas à me concentrer aussi bien que d'habitude",
                        2: "J'ai du mal à me concentrer longtemps sur quoi que ce soit",
                        3: "Je me trouve incapable de me concentrer sur quoi que ce soit"
                    },
                    "20. Fatigue": {
                        0: "Je ne suis pas plus fatigué(e) que d'habitude",
                        1: "Je me fatigue plus facilement que d'habitude",
                        2: "Je suis trop fatigué(e) pour faire un grand nombre de choses que je faisais avant",
                        3: "Je suis trop fatigué(e) pour faire la plupart des choses que je faisais avant"
                    },
                    "21. Perte d'intérêt pour le sexe": {
                        0: "Je n'ai pas noté de changement récent dans mon intérêt pour le sexe",
                        1: "Le sexe m'intéresse moins qu'avant",
                        2: "Le sexe m'intéresse beaucoup moins maintenant",
                        3: "J'ai perdu tout intérêt pour le sexe"
                    }
                }

            bdi_responses = {}

            with st.form("bdi_form"):
                st.markdown("""<div style='font-size: 16px;'>
                              <b>CONSIGNES</b>: Choisissez la réponse qui correspond le mieux à votre état au cours des <b>3 dernières semaines</b>.<br><br><br>
                              </div>""", unsafe_allow_html=True)

                # Display each question
                for idx, (question, choices_dict) in enumerate(bdi_ii_fr.items()):
                    options = list(choices_dict.values())
                    selected_label = st.radio(
                        label=question,
                        options=options,
                        index=None,
                        key=f"bdi_{idx}"
                    )
                    # Save the numeric value
                    bdi_responses[question] = [k for k, v in choices_dict.items() if v == selected_label][0] if selected_label is not None else None

                submit_bdi = st.form_submit_button("Soumettre")

                if submit_bdi:
                    if any(v is None for v in bdi_responses.values()):
                        st.warning("Veuillez répondre à toutes les questions avant de soumettre.")
                    else:
                        # Create DataFrame
                        bdi_df = pd.DataFrame([bdi_responses])

                        # Add participant info
                        bdi_df.insert(0, 'participant_id', st.session_state.participant_id)
                        bdi_df.insert(1, 'session_code', st.session_state.session_code)
                        bdi_df.insert(2, 'assessment_time', st.session_state.assessment_time)

                        # Compute total score
                        bdi_df.insert(3, 'total_score', bdi_df[list(bdi_ii_fr.keys())].astype(int).sum(axis=1))

                        # Save in session_state
                        st.session_state.bdi_results_df = bdi_df
                        st.success("Questionnaire BDI-II soumis avec succès !")

                        # Move to next page automatically
                        st.session_state.page = "EVA"
                        st.stop()

# --- Page 5: Visual Analogue Scales ---
elif page == "EVA":
    if not st.session_state.get("info_collected", False):
        st.warning("Veuillez d'abord remplir les informations du participant.")
    else:
        if "eva_results_df" not in st.session_state:
            st.session_state.eva_results_df = None

        st.markdown("""<div style='font-size: 32px; font-weight: bold;'>EVA</div>""", unsafe_allow_html=True)

        with st.form("eva_form"):

            st.markdown("""<div style='font-size: 16px;'> 
            <b>CONSIGNES</b>: Vous trouverez ci-dessous 3 questions concernant votre niveau de <b>fatigue</b>, de <b>tristesse</b> et d'<b>anxiété</b>. Merci d'indiquer sur la ligne horizontale ce qui représente le mieux votre état au <b>moment du remplissage</b>.<br><br><br>
            </div>""", unsafe_allow_html=True)

            # Sliders for each VAS
            fatigue = st.slider("Fatigue", min_value=0, max_value=100, value=50, step=1)
            sadness = st.slider("Tristesse", min_value=0, max_value=100, value=50, step=1)
            anxiety = st.slider("Anxiété", min_value=0, max_value=100, value=50, step=1)

            submit_vas = st.form_submit_button("Soumettre")

            if submit_vas:
                # Store data in a DataFrame
                eva_df = pd.DataFrame([{
                    "Fatigue": fatigue,
                    "Tristesse": sadness,
                    "Anxiété": anxiety,
                    "participant_id": st.session_state.participant_id,
                    "session_code": st.session_state.session_code,
                    "assessment_time": st.session_state.assessment_time
                }])

                st.session_state.eva_results_df = eva_df
                st.success("EVA soumis avec succès !")

                # Move to next page automatically
                st.session_state.page = "Téléchargement"
                st.stop()


# --- Page 6: Download data ---
elif page == "Téléchargement":
    st.markdown("""<div style='font-size: 32px; font-weight: bold;'>Téléchargement</div>""", unsafe_allow_html=True)

    # Define required questionnaires depending on assessment time
    if st.session_state.assessment_time == "Baseline (H:0)":
        required_keys = ["stai_results_df", "bdi_results_df", "eva_results_df"]
    else:  # Post-task (H:3)
        required_keys = ["stai_results_df", "eva_results_df"]

    all_completed = all(st.session_state.get(k) is not None for k in required_keys)

    if not all_completed:
        st.warning("Veuillez remplir tous les questionnaires requis avant de télécharger les résultats.")
        for k in required_keys:
            if st.session_state.get(k) is None:
                st.info(f"Résultats manquants pour : **{k.replace('_results_df', '').upper()}**")
    else:
        st.success("Tous les questionnaires requis ont été complétés. Vous pouvez maintenant télécharger les résultats.")

    # Define the dataframes to show
    if st.session_state.assessment_time == "Baseline (H:0)":
        dataframes = {
            "stai_results_df": "STAI-Y",
            "bdi_results_df": "BDI-II",
            "eva_results_df": "EVA",
        }
    else:  # Post-task (H:3)
        dataframes = {
            "stai_results_df": "STAI-Y",
            "eva_results_df": "EVA",
        }

    for key, label in dataframes.items():
        df = st.session_state.get(key)

        if df is not None:
            st.subheader(f"{label} Résultats")
            st.dataframe(df)

            # Show key scores below the dataframe
            if key == "stai_results_df":
                st.markdown("**Scores STAI-Y:**")
                if st.session_state.assessment_time == "Baseline (H:0)":
                    st.write(f"Score État : {df['score_state'].iloc[0]}")
                    st.write(f"Score Trait : {df['score_trait'].iloc[0]}")
                    st.write(f"Score Total : {df['score_total'].iloc[0]}")
                else:
                    st.write(f"Score État : {df['score_state'].iloc[0]}")
            elif key == "bdi_results_df":
                st.markdown("**Score BDI-II :**")
                st.write(f"{df['total_score'].iloc[0]}")
            elif key == "eva_results_df":
                st.markdown("**Scores EVA :**")
                st.write(f"Fatigue : {df['Fatigue'].iloc[0]}")
                st.write(f"Tristesse : {df['Tristesse'].iloc[0]}")
                st.write(f"Anxiété : {df['Anxiété'].iloc[0]}")

            # Download button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Télécharger {label}.csv",
                data=csv_data,
                file_name=f"{st.session_state.participant_id}_{st.session_state.session_code}_{label}_{st.session_state.assessment_time}.csv",
                mime='text/csv'
            )
        else:
            # Only show missing info if it was expected
            if key in required_keys:
                st.info(f"Aucun résultat disponible pour {label}.")
