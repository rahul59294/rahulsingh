# IPL Fantasy Model in R
# Required packages: dplyr, readr, xgboost, caret, ompr, ompr.roi, ROI.plugin.glpk

library(dplyr)
library(readr)
library(xgboost)
library(caret)
library(ompr)
library(ompr.roi)
library(ROI.plugin.glpk)

# -------------------------------
# 1. LOAD DATA
# -------------------------------

batting <- read_csv("bowling_data.csv")
bowling <- read_csv("batting_data.csv")
fielding <- read_csv("fielding_data.csv")
dismissals = read_csv("dismissals.csv")
partnership_by_wicket = read_csv("partnership_by_wickets.csv")
partnership_by_runs = read.csv("partnership_by_runs.csv")
Batting_scores = read.csv("Batting_scores.csv")
bowling_scores = read_csv("bowling_scores.csv")
fielding_scores = read_csv("fielding_scores.csv")
SR_scores = read_csv("SR_scores.csv")
Economy_scores = read_csv("Economy_scores.csv")
other_score = read_csv("other_score.csv")
RCB_batting = read.csv("RCB_batting.csv")
RCB_bowling = read.csv("RCB_bowling.csv")
KKR_Batting = read.csv("KKR_Batting.csv")
KKR_Bowling = read.csv("KKR_Bowling.csv")
RR_Batting = read_csv("RR_Batting.csv")
RR_Bowling = read.csv("RR_Bowling.csv")
SRH_Batting = read_csv("SRH_Batting.csv")
SRH_Bowling = read.csv("SRH_Bowling.csv")
MI_Batting = read.csv("MI_Batting.csv")
MI_Bowling = read.csv("MI_Bowling.csv")
CSK_Batting = read.csv("CSK_Batting.csv")
CSK_Bowling = read.csv("CSK_Bowling.csv")
DC_Batting = read_csv("DC_Batting.csv")
DC_Bowling = read.csv("DC_Bowling.csv")
LSG_Batting = read_csv("LSG_Batting.csv")
LSG_Bowling = read.csv("LSG_Bowling.csv")
GT_Batting = read_csv("GT_Batting.csv")
GT_Bowling = read.csv("GT_Bowling.csv")
PBKS_Batting = read_csv("PBKS_Batting.csv")
PBKS_Bowling = read.csv("PBKS_Bowling.csv")

# -------------------------------
# 2. APPLY SCORING SYSTEM
# -------------------------------

get_score <- function(row, scoring) {
  score <- 0
  score <- score + row$Runs  * scoring$points[scoring$event == "run"]
  score <- score + row$`4s`  * scoring$points[scoring$event == "four"]
  score <- score + row$`6s`  * scoring$points[scoring$event == "six"]
  score <- score + row$wkts * scoring$points[scoring$event == "wicket"]
  score <- score + row$ct    * scoring$points[scoring$event == "catch"]
  return(score)
}

batting <- batting %>%
  rowwise() %>%
  mutate(fantasy_points = get_score(cur_data(), scoring))

# -------------------------------
# 3. TRAIN MODEL
# -------------------------------

merged_df <- batting %>% 
  filter(!is.na(fantasy_points)) %>%
  select(player = players, Runs, `4s`, `6s`, wkts, ct, fantasy_points)

train_matrix <- as.matrix(merged_df[, -c(1, ncol(merged_df))])
train_label <- merged_df$fantasy_points

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)

# -------------------------------
# 4. PREDICT PLAYER SCORES
# -------------------------------

upcoming_players <- batting %>% filter(year == 2025) %>%
  select(player = players, Runs, `4s`, `6s`, wkts, ct)

upcoming_matrix <- as.matrix(upcoming_players[, -1])
upcoming_players$exp_points <- predict(model, upcoming_matrix)

# Add mock data for demo
set.seed(1)
upcoming_players <- upcoming_players %>%
  mutate(role = sample(c("BAT", "BWL", "AR", "WK"), n(), replace = TRUE),
         ipl_team = sample(c("RCB", "CSK", "MI", "KKR"), n(), replace = TRUE),
         credits = runif(n(), 7, 10))

# -------------------------------
# 5. OPTIMISE TEAM
# -------------------------------

N <- nrow(upcoming_players)

model_ilp <- MIPModel() %>%
  add_variable(x[i], i = 1:N, type = "binary") %>%
  set_objective(sum_expr(x[i] * upcoming_players$exp_points[i], i = 1:N), "max") %>%
  add_constraint(sum_expr(x[i], i = 1:N) == 11) %>%
  add_constraint(sum_expr(x[i] * upcoming_players$credits[i], i = 1:N) <= 100) %>%
  add_constraint(sum_expr(x[i] * (upcoming_players$role[i] == "WK"), i = 1:N) >= 1) %>%
  add_constraint(sum_expr(x[i] * (upcoming_players$role[i] == "BAT"), i = 1:N) >= 3) %>%
  add_constraint(sum_expr(x[i] * (upcoming_players$role[i] == "BWL"), i = 1:N) >= 3) %>%
  add_constraint(sum_expr(x[i] * (upcoming_players$role[i] == "AR"), i = 1:N) >= 1)

result <- solve_model(model_ilp, with_ROI(solver = "glpk", verbose = TRUE))

# -------------------------------
# 6. OUTPUT FINAL XI
# -------------------------------

chosen_indices <- which(get_solution(result, x[i])$value == 1)
final_team <- upcoming_players[chosen_indices, ]

print(final_team %>% select(player, exp_points, credits, role, ipl_team))
