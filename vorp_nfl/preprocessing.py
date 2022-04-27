import pandas as pd
import logging

logger = logging.getLogger(__file__)

def extract_player(df):
    # If play-type is pass: then need to get the QB and the reciever
    # If play-type is rush, then need to get RB
    # If play-type is no play, then ignore

    descs = df["Description"].values
    isrush = df['IsRush'].values
    ispass = df["IsPass"].values
    oteams = df["OffenseTeam"].values

    qbs = []
    rbs = []
    wrs = []

    for i in range(len(df)):

        if isrush[i] == 1:
            splits = descs[i].split("-")

            if splits[0][-1].isdigit():
                if splits[0][-2].isdigit():
                    num = splits[0][-2:]
                else:
                    num = splits[0][-1]
            else:
                logger.warning("Row: {} not formatted correctly. Desc: {}".format(i, descs[i]))
                rbs.append(None)
                wrs.append(None)
                qbs.append(None)
                continue

            oteam = oteams[i]
            player = "{}-{}".format(oteam, num)
            rbs.append(player)
            wrs.append(None)
            qbs.append(None)

        elif ispass[i] == 1:
            splits = descs[i].split("-")

            if splits[0][-1].isdigit():
                if splits[0][-2].isdigit():
                    qb_num = splits[0][-2:]
                else:
                    qb_num = splits[0][-1]
            else:
                logger.warning("Row: {} not formatted correctly. Desc: {}".format(i, descs[i]))
                rbs.append(None)
                wrs.append(None)
                qbs.append(None)
                continue

            if splits[1][-1].isdigit():
                if splits[1][-2].isdigit():
                    wr_num = splits[1][-2:]
                else:
                    wr_num = splits[1][-1]
            else:
                logger.warning("Row: {} possibly not formatted correctly. Desc: {}. Adding QB anyway".format(i, descs[i]))
                wr_num = None

            oteam = oteams[i]
            qb = "{}-{}".format(oteam, qb_num)

            if wr_num is None:
                wr = None
            else:
                wr = "{}-{}".format(oteam, wr_num)

            rbs.append(None)
            wrs.append(wr)
            qbs.append(qb)

        else:
            rbs.append(None)
            qbs.append(None)
            wrs.append(None)

    df["QB"] = qbs
    df["WR"] = wrs
    df["RB"] = rbs


def create_time_features(df):

    df["GameDate"] = pd.to_datetime(df["GameDate"])
    min_date = df["GameDate"].min()

    df["DateInt"] = df["GameDate"] - min_date
    df["DateInt"] = df["DateInt"].apply(lambda x: x.days)

    quarter = df["Quarter"]-1
    quarter = quarter.values
    game_time = quarter*15 + df["Minute"] + df["Second"]/60
    df["GameTime"] = game_time


if __name__ == "__main__":
    df = pd.read_csv("/Users/jackarmand/Documents/GitHub/vorp-nfl/vorp_nfl/data/pbp-2021.csv")
    sdf = df.iloc[:1000]
    extract_player(df)
    print(df[["QB", "WR", "RB"]])