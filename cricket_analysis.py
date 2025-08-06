# main.py
import pandas as pd
import numpy as np
import re
import warnings
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 0. HELPER FUNCTIONS & INITIAL SETUP
# =============================================================================

def safe_num(series):
    return pd.to_numeric(series, errors='coerce')

def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.replace(u'\xa0', ' ')
    name = re.sub(r"[^a-zA-Z0-9 .'-]", "", name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()

def standardise_batting(df, team_code):
    df['team'] = team_code
    if 'SR' not in df.columns:
        df['SR'] = np.nan
    possible_cols = ["R", "B", "4s", "6s", "SR", "Score"]
    for col in possible_cols:
        if col in df.columns:
            df[col] = safe_num(df[col])
    return df

def standardise_bowling(df, team_code):
    df['team'] = team_code
    possible_cols = ["R", "O", "M", "W", "Econ", "Dots", "Score"]
    for col in possible_cols:
        if col in df.columns:
            df[col] = safe_num(df[col])
    return df

# =============================================================================
# 1. LOAD DATASETS
# This section assumes all your CSV files are in the same directory as the script.
# =============================================================================
print("Step 1: Loading and Standardizing Data...")

try:
    batting_data = pd.read_csv("Batting_data.csv")
    bowling_data = pd.read_csv("Bowling_data.csv")
    fielding_data = pd.read_csv("fielding_data.csv")
    dismissals_data = pd.read_csv("dismissals.csv")
    partnership_runs_data = pd.read_csv("partnership_by_runs.csv")
    match_data = pd.read_csv("match_data.csv")

    batting_order_df = pd.read_csv("batting_order.csv")
    bowling_order_df = pd.read_csv("bowling_order.csv")

    batting_order_df['Player'] = batting_order_df['Player'].apply(normalize_name)
    bowling_order_df['Player'] = bowling_order_df['Player'].apply(normalize_name)

    RCB_batting = standardise_batting(pd.read_csv("RCB_batting.csv"), "RCB")
    KKR_Batting = standardise_batting(pd.read_csv("KKR_Batting.csv"), "KKR")
    RR_Batting = standardise_batting(pd.read_csv("RR_Batting.csv"), "RR")
    SRH_Batting = standardise_batting(pd.read_csv("SRH_Batting.csv"), "SRH")
    MI_Batting = standardise_batting(pd.read_csv("MI_Batting.csv"), "MI")
    CSK_Batting = standardise_batting(pd.read_csv("CSK_Batting.csv"), "CSK")
    DC_Batting = standardise_batting(pd.read_csv("DC_Batting.csv"), "DC")
    LSG_Batting = standardise_batting(pd.read_csv("LSG_Batting.csv"), "LSG")
    GT_Batting = standardise_batting(pd.read_csv("GT_Batting.csv"), "GT")
    PBKS_Batting = standardise_batting(pd.read_csv("PBKS_Batting.csv"), "PBKS")

    RCB_bowling = standardise_bowling(pd.read_csv("RCB_bowling.csv"), "RCB")
    KKR_Bowling = standardise_bowling(pd.read_csv("KKR_Bowling.csv"), "KKR")
    RR_Bowling = standardise_bowling(pd.read_csv("RR_Bowling.csv"), "RR")
    SRH_Bowling = standardise_bowling(pd.read_csv("SRH_Bowling.csv"), "SRH")
    MI_Bowling = standardise_bowling(pd.read_csv("MI_Bowling.csv"), "MI")
    CSK_Bowling = standardise_bowling(pd.read_csv("CSK_Bowling.csv"), "CSK")
    DC_Bowling = standardise_bowling(pd.read_csv("DC_Bowling.csv"), "DC")
    LSG_Bowling = standardise_bowling(pd.read_csv("LSG_Bowling.csv"), "LSG")
    GT_Bowling = standardise_bowling(pd.read_csv("GT_Bowling.csv"), "GT")
    PBKS_Bowling = standardise_bowling(pd.read_csv("PBKS_Bowling.csv"), "PBKS")

except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure all required CSV files are in the correct directory.")
    exit()

# =============================================================================
# 2. MERGE 2025 BATTING & BOWLING DATA
# =============================================================================
print("\nStep 2: Merging 2025 player data...")

all_batting_dfs = [
    RCB_batting, KKR_Batting, RR_Batting, SRH_Batting,
    MI_Batting, CSK_Batting, DC_Batting, LSG_Batting,
    GT_Batting, PBKS_Batting
]
batting_2025 = pd.concat(all_batting_dfs, ignore_index=True)

all_bowling_dfs = [
    RCB_bowling, KKR_Bowling, RR_Bowling, SRH_Bowling,
    MI_Bowling, CSK_Bowling, DC_Bowling, LSG_Bowling,
    GT_Bowling, PBKS_Bowling
]
bowling_2025 = pd.concat(all_bowling_dfs, ignore_index=True)

# =============================================================================
# 3. COMPUTE 2025 FANTASY SCORES
# =============================================================================
print("\nStep 3: Calculating 2025 fantasy scores...")

batting_2025['Batter'] = batting_2025['Batter'].apply(normalize_name)
batting_2025['team'] = batting_2025['team'].str.strip()
bowling_2025['Bowler'] = bowling_2025['Bowler'].apply(normalize_name)
bowling_2025['team'] = bowling_2025['team'].str.strip()
dismissals_data['Player'] = dismissals_data['Player'].apply(normalize_name)

batting_2025_scored_clean = batting_2025.groupby(['Batter', 'team'])['Score'].mean().reset_index()
batting_2025_scored_clean = batting_2025_scored_clean.rename(columns={'Score': 'batting_score'})

bowling_2025_scored_clean = bowling_2025.groupby(['Bowler', 'team'])['Score'].mean().reset_index()
bowling_2025_scored_clean = bowling_2025_scored_clean.rename(columns={'Score': 'bowling_score', 'Bowler': 'Batter'})

fantasy_scores_2025 = pd.merge(
    batting_2025_scored_clean,
    bowling_2025_scored_clean,
    on='Batter',
    how='outer',
    suffixes=('_bat', '_bowl')
)

fantasy_scores_2025['team'] = fantasy_scores_2025['team_bat'].fillna(fantasy_scores_2025['team_bowl'])
fantasy_scores_2025 = fantasy_scores_2025.drop(columns=['team_bat', 'team_bowl'])

fantasy_scores_2025['batting_score'] = fantasy_scores_2025['batting_score'].fillna(0)
fantasy_scores_2025['bowling_score'] = fantasy_scores_2025['bowling_score'].fillna(0)
fantasy_scores_2025['fantasy_score'] = fantasy_scores_2025['batting_score'] + fantasy_scores_2025['bowling_score']

conditions = [
    (fantasy_scores_2025['batting_score'] > 0) & (fantasy_scores_2025['bowling_score'] > 0),
    (fantasy_scores_2025['batting_score'] > 0),
    (fantasy_scores_2025['bowling_score'] > 0)
]
choices = ['Allrounder', 'Batter', 'Bowler']
fantasy_scores_2025['Role'] = np.select(conditions, choices, default='Other')
fantasy_scores_2025 = fantasy_scores_2025.rename(columns={'Batter': 'Player'})

wicketkeepers = dismissals_data[dismissals_data['St'] > 0]['Player'].unique()
fantasy_scores_2025['is_wicketkeeper'] = fantasy_scores_2025['Player'].isin(wicketkeepers)

def update_role(row):
    if row['is_wicketkeeper'] and row['Role'] == 'Batter':
        return 'Wicketkeeper'
    return row['Role']

fantasy_scores_2025['Role'] = fantasy_scores_2025.apply(update_role, axis=1)

fantasy_scores_2025 = fantasy_scores_2025.merge(batting_order_df[['Player', 'Order']], on='Player', how='left')
fantasy_scores_2025 = fantasy_scores_2025.rename(columns={'Order': 'batting_order'})

fantasy_scores_2025 = fantasy_scores_2025.merge(bowling_order_df[['Player', 'Primary Overs']], on='Player', how='left')
fantasy_scores_2025 = fantasy_scores_2025.rename(columns={'Primary Overs': 'bowling_order'})

# =============================================================================
# 4. BUILDING A DIVERSE TEAM (Initial selection)
# =============================================================================
print("\nStep 4: Building an initial diverse fantasy team...")

role_requirements = {'Batter': 4, 'Bowler': 3, 'Allrounder': 2, 'Flex': 2}

def select_players_by_role(scores_df, role, count):
    """Selects top players for a given role based on score."""
    return scores_df[scores_df['Role'] == role].sort_values(by='fantasy_score', ascending=False).head(count)

# Select players based on role requirements
batters = select_players_by_role(fantasy_scores_2025, 'Batter', role_requirements['Batter'])
bowlers = select_players_by_role(fantasy_scores_2025, 'Bowler', role_requirements['Bowler'])
allrounders = select_players_by_role(fantasy_scores_2025, 'Allrounder', role_requirements['Allrounder'])

# Combine selected players
selected_players = pd.concat([batters, bowlers, allrounders])

# Select flex players from the remaining pool
remaining_players = fantasy_scores_2025[~fantasy_scores_2025['Player'].isin(selected_players['Player'])]
flex_players = remaining_players.sort_values(by='fantasy_score', ascending=False).head(role_requirements['Flex'])

# Final diverse team
fantasy_team_diverse = pd.concat([selected_players, flex_players])
print("Initial Diverse Fantasy XI:")
print(fantasy_team_diverse[['Player', 'team', 'Role', 'fantasy_score']])

# =============================================================================
# 4B. PLAYER CONSISTENCY (2025)
# =============================================================================
print("\nStep 4B: Calculating player consistency...")

batting_consistency = batting_2025[['Batter', 'team', 'Score']].rename(columns={'Batter': 'Player'})
bowling_consistency = bowling_2025[['Bowler', 'team', 'Score']].rename(columns={'Bowler': 'Player'})

consistency_data = pd.concat([batting_consistency, bowling_consistency])

# Calculate consistency metrics (equivalent to group_by %>% summarise)
consistency_summary = consistency_data.groupby('Player').agg(
    matches_played=('Score', 'nunique'),
    avg_score=('Score', 'mean'),
    sd_score=('Score', 'std')
).reset_index()

consistency_summary['sd_score'] = consistency_summary['sd_score'].fillna(0)
consistency_summary['consistency_index'] = np.where(
    consistency_summary['matches_played'] >= 5,
    consistency_summary['avg_score'] / (consistency_summary['sd_score'] + 1e-6),
    np.nan
)

consistency_summary = consistency_summary.sort_values(
    by=['matches_played', 'consistency_index'], ascending=[False, False]
)
print("\nTop 15 Most Consistent Players (2025):")
print(consistency_summary.head(15))


# =============================================================================
# 5. PROCESS HISTORICAL DATA & CREATE HISTORICAL SCORE
# =============================================================================
print("\nStep 5: Processing historical data to create a historical score...")

# Normalize names
for df in [batting_data, bowling_data, fielding_data, dismissals_data]:
    df['Player'] = df['Player'].apply(normalize_name)

# Summarize features for each domain
batting_features = batting_data.groupby('Player').agg(
    total_runs=('Runs', 'sum'), avg_runs=('Runs', 'mean'),
    strike_rate=('SR', 'mean'), innings=('Player', 'size'),
    fours=('4s', 'sum'), sixes=('6s', 'sum'),
    fifties=('50', 'sum'), hundreds=('100', 'sum')
).reset_index()

bowling_features = bowling_data.groupby('Player').agg(
    total_wickets=('Wkts', 'sum'), avg_economy=('Econ', 'mean'),
    bowling_SR=('SR', 'mean'), matches=('Player', 'size')
).reset_index()

fielding_features = fielding_data.groupby('Player').agg(
    total_catches=('Ct', 'sum'), avg_ct_inn=('Ct/Inn', 'mean')
).reset_index()

dismissals_data['dismissed'] = safe_num(dismissals_data['Dismissed'])
dismissals_data['St'] = safe_num(dismissals_data['St'])
dismissals_data['catches_dismissals'] = safe_num(dismissals_data['Ct'])
dismissals_features = dismissals_data.groupby('Player').agg(
    total_dismissals=('dismissed', 'sum'),
    stumps=('St', 'sum'),
    catches=('catches_dismissals', 'sum')
).reset_index()

# Partnership synergy (equivalent to pivot_longer and summarise)
partnership_runs_data['Runs'] = safe_num(partnership_runs_data['Runs'])
good_partnerships_raw = partnership_runs_data[partnership_runs_data['Runs'] >= 50]
good_partnerships_melted = good_partnerships_raw.melt(
    id_vars=['Runs'], value_vars=['Player1', 'Player2'], value_name='Player'
)
good_partnerships_melted['Player'] = good_partnerships_melted['Player'].apply(normalize_name)
good_partnerships = good_partnerships_melted.groupby('Player').size().reset_index(name='good_partnerships')

# Merge all historical features
historical_features = pd.merge(batting_features, bowling_features, on='Player', how='outer')
historical_features = pd.merge(historical_features, fielding_features, on='Player', how='outer')
historical_features = pd.merge(historical_features, dismissals_features, on='Player', how='outer')
historical_features = pd.merge(historical_features, good_partnerships, on='Player', how='outer')
historical_features = historical_features.fillna(0)

# Calculate historical score
historical_features['batting_score_hist'] = (0.4 * historical_features['avg_runs'] + 0.2 * historical_features['strike_rate'] +
                                             0.1 * historical_features['fifties'] + 0.1 * historical_features['hundreds'])
historical_features['bowling_score_hist'] = (0.3 * historical_features['total_wickets'] - 0.1 * historical_features['avg_economy'] +
                                             0.1 * historical_features['matches'])
historical_features['fielding_score_hist'] = (0.2 * historical_features['total_catches'] + 0.1 * historical_features['avg_ct_inn'])
historical_features['dismissal_score_hist'] = (0.1 * historical_features['total_dismissals'] + 0.05 * historical_features['stumps'] +
                                               0.05 * historical_features['catches'])
historical_features['partnership_score_hist'] = 0.1 * historical_features['good_partnerships']

score_cols = ['batting_score_hist', 'bowling_score_hist', 'fielding_score_hist', 'dismissal_score_hist', 'partnership_score_hist']
historical_features['historical_score'] = historical_features[score_cols].sum(axis=1)

print("\nTop 15 Players by Historical Score:")
print(historical_features[['Player', 'historical_score']].sort_values(by='historical_score', ascending=False).head(15))

# =============================================================================
# 6. FILTER BY MATCH CONTEXT & GENERATE FINAL TEAM (with dropdowns in sidebar)
# =============================================================================
st.title("IPL Fantasy XI Selector")

required_vars = ['fantasy_scores_2025', 'match_data', 'consistency_summary', 'historical_features', 'role_requirements']
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    st.error(f"Missing variables: {', '.join(missing_vars)}. Please ensure all sections above are properly executed.")
else:
    with st.sidebar:
        st.header("Select Match Context")
        teams_list = sorted(fantasy_scores_2025['team'].dropna().unique())
        venue_list = sorted(match_data['Ground'].dropna().unique())

        user_team1 = st.selectbox("Team 1", teams_list, index=teams_list.index("MI") if "MI" in teams_list else 0)
        user_team2 = st.selectbox("Team 2", teams_list, index=teams_list.index("CSK") if "CSK" in teams_list else 1)
        user_venue = st.selectbox("Venue", venue_list, index=venue_list.index("Wankhede") if "Wankhede" in venue_list else 0)

        st.header("Select Team Type Preference")
        team_type = st.selectbox("Preferred Team Type", ["Balanced (Default)", "High Scorers", "Consistent Performers"])

        if team_type == "High Scorers":
            weights = {"fantasy": 0.7, "consistency": 0.1, "historical": 0.2}
        elif team_type == "Consistent Performers":
            weights = {"fantasy": 0.3, "consistency": 0.5, "historical": 0.2}
        else:
            weights = {"fantasy": 0.5, "consistency": 0.2, "historical": 0.3}

    if user_team1 == user_team2:
        st.warning("Team 1 and Team 2 must be different.")
    else:
        relevant_players_context = fantasy_scores_2025[fantasy_scores_2025['team'].isin([user_team1, user_team2])]

        fantasy_scores_contextual = pd.merge(relevant_players_context, consistency_summary, on='Player', how='left')
        fantasy_scores_contextual = pd.merge(fantasy_scores_contextual, historical_features, on='Player', how='left')

        fantasy_scores_contextual['consistency_index'] = fantasy_scores_contextual['consistency_index'].fillna(0)
        fantasy_scores_contextual['historical_score'] = fantasy_scores_contextual['historical_score'].fillna(0)

        fantasy_scores_contextual['combined_score'] = (
            weights['fantasy'] * fantasy_scores_contextual['fantasy_score'] +
            weights['consistency'] * fantasy_scores_contextual['consistency_index'] +
            weights['historical'] * fantasy_scores_contextual['historical_score']
        )

        # Remove either top-order batsmen or powerplay bowlers based on average score
        top_order_batsmen = fantasy_scores_contextual[fantasy_scores_contextual['batting_order'] <= 3]
        powerplay_bowlers = fantasy_scores_contextual[fantasy_scores_contextual['bowling_order'] == 'Powerplay']

        top_batsmen_score = top_order_batsmen['combined_score'].mean() if not top_order_batsmen.empty else 0
        power_bowlers_score = powerplay_bowlers['combined_score'].mean() if not powerplay_bowlers.empty else 0

        if top_batsmen_score >= power_bowlers_score:
            fantasy_scores_contextual = fantasy_scores_contextual[
                ~((fantasy_scores_contextual['bowling_order'] == 'Powerplay') & (fantasy_scores_contextual['batting_order'] <= 3))]
        else:
            fantasy_scores_contextual = fantasy_scores_contextual[
                ~((fantasy_scores_contextual['batting_order'] <= 3) & (fantasy_scores_contextual['bowling_order'] == 'Powerplay'))]

        def get_valid_team_pool():
            pool = fantasy_scores_contextual.copy()
            wk = pool[pool['Role'] == 'Wicketkeeper'].sort_values(by='combined_score', ascending=False).head(1)
            pool = pool[~((pool['Role'] == 'Wicketkeeper') & (~pool['Player'].isin(wk['Player'])))]
            return pool

        pool = get_valid_team_pool()

        def generate_team(pool):
            role_requirements = {'Batter': 4, 'Bowler': 3, 'Allrounder': 2, 'Flex': 1}
            batters = pool[pool['Role'] == 'Batter'].nlargest(role_requirements['Batter'], 'combined_score')
            bowlers = pool[pool['Role'] == 'Bowler'].nlargest(role_requirements['Bowler'], 'combined_score')
            allrounders = pool[pool['Role'] == 'Allrounder'].nlargest(role_requirements['Allrounder'], 'combined_score')
            wicketkeeper = pool[pool['Role'] == 'Wicketkeeper'].head(1)

            selected = pd.concat([batters, bowlers, allrounders, wicketkeeper])
            remaining = pool[~pool['Player'].isin(selected['Player'])]
            flex = remaining.nlargest(role_requirements['Flex'], 'combined_score')
            final_team = pd.concat([selected, flex]).drop_duplicates('Player')
            return final_team

        team = generate_team(pool)

        def enforce_team_min(team):
            team_counts = team['team'].value_counts().to_dict()
            for t in [user_team1, user_team2]:
                if team_counts.get(t, 0) < 4:
                    needed = 4 - team_counts.get(t, 0)
                    add = pool[(pool['team'] == t) & (~pool['Player'].isin(team['Player']))].nlargest(needed, 'combined_score')
                    team = pd.concat([team, add]).drop_duplicates('Player').nlargest(11, 'combined_score')
            return team

        team = enforce_team_min(team)

        roles_needed = ['Batter', 'Bowler', 'Allrounder', 'Wicketkeeper']
        if not all(any((team['Role'] == r) & (team['team'].isin([user_team1, user_team2]))) for r in roles_needed):
            st.error("Final team does not meet minimum role requirements.")
        else:
            team = team.sort_values(by='combined_score', ascending=False).reset_index(drop=True)
            team['Multiplier'] = 1.0
            team.at[0, 'Multiplier'] = 2.0
            team.at[1, 'Multiplier'] = 1.5
            team['final_score'] = team['combined_score'] * team['Multiplier']
            total_score = team['final_score'].sum()

            st.subheader(f"Fantasy XI for {user_team1} vs {user_team2}")
            st.dataframe(team[['Player', 'team', 'Role', 'batting_order', 'bowling_order' , 'fantasy_score', 'consistency_index', 'historical_score', 'combined_score', 'Multiplier', 'final_score']])
            st.markdown(f"**Total Team Score (after multipliers):** {total_score:.2f}")
            st.bar_chart(team.set_index('Player')['final_score'])
            csv = team.to_csv(index=False).encode('utf-8')
            st.download_button("Download Fantasy XI as CSV", csv, file_name="fantasy_team.csv", mime="text/csv")

