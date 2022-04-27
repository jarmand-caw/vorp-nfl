# vorp-nfl

### Filters

- IsTwoPointConversion == 0
- IsPenalty == 0
- IsFumble == 0
- IsInterception == 0
- IsSack == 0
- PlayType: Separate models for both pass and rush

### Features

Categorical
- Quarter
- OffenseTeam
- DefenseTeam
- Down
- SeriesFirstDown
- Formation
- Passing Only:
  - PassType
- Rushing Only:
  - RushDirection

Continuous
- ToGo
- YardLine
- DateInt
- GameTime

Target
- Yards
