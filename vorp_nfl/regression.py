from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.simplefilter("ignore")

logger = logging.getLogger(__file__)


class VORP(object):
    def __init__(
            self,
            df,
            label=None,
            continuous_features=None,
            categorical_features=None,
            passing_features=None,
            rushing_features=None
    ):

        self.df = df
        self.df = self.df.loc[self.df["IsTwoPointConversion"] == 0]
        self.df = self.df.loc[self.df["IsFumble"] == 0]
        self.df = self.df.loc[self.df["IsInterception"] == 0]
        self.df = self.df.loc[self.df["IsSack"] == 0]

        self.continuous_features = continuous_features if continuous_features is not None else ["ToGo", "YardLine", "DateInt", "GameTime"]
        self.categorical_features = categorical_features if categorical_features is not None else ["OffenseTeam", "DefenseTeam", "Down", "SeriesFirstDown", "Formation"]
        self.label = label if label is not None else ["Yards"]

        self.passing_features = passing_features if passing_features is not None else ["PassType"]
        self.rushing_features = rushing_features if rushing_features is not None else ["RushDirection"]

        self.category_map = {}
        for feature in self.categorical_features+self.passing_features+self.rushing_features:
            options = list(set(self.df[feature]))
            self.category_map[feature] = dict(zip(options, [x for x in range(len(options))]))
            self.df[feature] = self.df[feature].map(self.category_map[feature])

        self.rush_df = self.df.loc[self.df["PlayType"] == "RUSH"]
        self.rush_df.drop(["PlayType"], axis=1, inplace=True)

        self.pass_df = self.df.loc[self.df["PlayType"] == "PASS"]
        self.pass_df.drop(["PlayType"], axis=1, inplace=True)

        self.train_rush_df, self.test_rush_df = train_test_split(self.rush_df, test_size=0.3, shuffle=True, random_state=42)
        self.train_rush_df, self.val_rush_df = train_test_split(self.train_rush_df, test_size=0.1, shuffle=True, random_state=42)

        self.train_pass_df, self.test_pass_df = train_test_split(self.pass_df, test_size=0.3, shuffle=True, random_state=42)
        self.train_pass_df, self.val_pass_df = train_test_split(self.train_pass_df, test_size=0.1, shuffle=True, random_state=42)

    def create_xy(self, df, type, ignore_features=None):

        ignore_features = ignore_features if ignore_features is not None else []

        df = df.copy()

        if type == "pass":
            categorical_features = self.categorical_features + self.passing_features
        elif type == "rush":
            categorical_features = self.categorical_features + self.rushing_features
        else:
            raise TypeError("type must be in [pass, rush]. Is: {}".format(type))

        X_vars = [x for x in categorical_features+self.continuous_features if x not in ignore_features]
        X, y = df[X_vars], df[self.label]

        return X, y

    def train_rush_model(self, params=None, ignore_features=None, save_model=True):

        ignore_features = ignore_features if ignore_features is not None else []
        params = params if params is not None else {}
        params['objective'] = 'regression'

        X_train, y_train = self.create_xy(self.train_rush_df, type="rush", ignore_features=ignore_features)
        X_val, y_val = self.create_xy(self.val_rush_df, type="rush", ignore_features=ignore_features)
        X_test, y_test = self.create_xy(self.test_rush_df, type="rush", ignore_features=ignore_features)
        model = LGBMRegressor(**params)
        bst = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=15, verbose=False)
        ypred_test = bst.predict(X_test, num_iteration=bst.best_iteration_)
        ypred_train = bst.predict(X_train, num_iteration=bst.best_iteration_)

        logger.info("Train R2 Score: {}".format(r2_score(self.train_rush_df[self.label], ypred_train)))
        logger.info("Test R2 Score: {}".format(r2_score(self.test_rush_df[self.label], ypred_test)))
        logger.info(dict(zip(bst.feature_name_, bst.feature_importances_)))

        if save_model:
            X, y = self.create_xy(self.rush_df, type="rush", ignore_features=ignore_features)
            ypred = bst.predict(X, num_iteration=bst.best_iteration_)

            self.rush_df["Predicted Value"] = ypred
            self.rush_model = bst
        else:
            return bst


    def rush_regression(self, reg_var, params=None, ignore_features=None):

        ignore_features = ignore_features if ignore_features is not None else []
        params = params if params is not None else {}

        bst = self.train_rush_model(params, ignore_features, save_model=False)

        X_bst, y_bst = self.create_xy(self.rush_df, type="rush", ignore_features=ignore_features)
        ypred = bst.predict(X_bst, num_iteration=bst.best_iteration_)

        rush_df = self.rush_df.copy()
        rush_df["Predicted Value"] = ypred

        X = rush_df[["Predicted Value", reg_var]]
        y = rush_df[self.label]

        ols = LinearRegression()
        model = ols.fit(X, y)
        r2 = model.score(X, y)

        logger.info("Starting R2: {}".format(r2_score(y, rush_df["Predicted Value"])))
        logger.info("Regression R2 Score: {}".format(r2))
        logger.info(dict(zip(["Predicted Value", reg_var], list(model.coef_[0]))))

    def rush_player_basic_vorp(self):

        if "Predicted Value" not in list(self.rush_df.columns):
            raise NotImplementedError("Must run self.train_rush_model with save_model=True first")

        rush_df = pd.get_dummies(self.rush_df, prefix="RB", columns=["RB"])

        X_vars = ["Predicted Value"] + [x for x in rush_df if "RB" in x]
        X = rush_df[X_vars]
        y = rush_df[self.label]

        ols = Ridge(alpha=1e-4, normalize=True)
        model = ols.fit(X, y)
        r2 = model.score(X, y)

        logger.info("Starting R2: {}".format(r2_score(y, rush_df["Predicted Value"])))
        logger.info("Regression R2 Score: {}".format(r2))

        coef_df = pd.DataFrame(dict(zip(X_vars, [[x] for x in list(model.coef_[0])]))).transpose()
        coef_df.columns = ["COEF"]

        rb_counts = pd.DataFrame(rush_df[[x for x in rush_df if "RB" in x]].sum())
        rb_counts.columns = ["COUNT"]
        coef_df = pd.concat([coef_df, rb_counts], axis=1)

        coef_df = coef_df.loc[coef_df["COUNT"]>100]
        logger.info(coef_df.sort_values("COEF"))

        scaler = StandardScaler()
        coef_df["COEF"] = scaler.fit_transform(coef_df["COEF"].values.reshape(-1,1))
        coef_df = coef_df.sort_values("COEF")

        self.rush_basic_df = coef_df
        coef_df.to_csv(r"/Users/jackarmand/Documents/GitHub/vorp-nfl/vorp_nfl/data/rb_vorp.csv")

    def rush_player_multiplier_vorp(self):

        if "Predicted Value" not in list(self.rush_df.columns):
            raise NotImplementedError("Must run self.train_rush_model with save_model=True first")

        rush_df = pd.get_dummies(self.rush_df, prefix="RB", columns=["RB"])
        rb_counts = pd.DataFrame(rush_df[[x for x in rush_df if "RB" in x]].sum())
        rb_cols = [x for x in rush_df if "RB" in x]
        for col in rb_cols:
            rush_df[col] = rush_df[col].values * rush_df["Predicted Value"].values

        X_vars = rb_cols
        X = rush_df[X_vars]
        y = rush_df[self.label]

        ols = Ridge(alpha=1e-4, normalize=True)
        model = ols.fit(X, y)
        r2 = model.score(X, y)

        logger.info("Starting R2: {}".format(r2_score(y, rush_df["Predicted Value"])))
        logger.info("Regression R2 Score: {}".format(r2))

        coef_df = pd.DataFrame(dict(zip(X_vars, [[x] for x in list(model.coef_[0])]))).transpose()
        coef_df.columns = ["COEF"]

        rb_counts.columns = ["COUNT"]
        coef_df = pd.concat([coef_df, rb_counts], axis=1)

        coef_df = coef_df.loc[coef_df["COUNT"] > 100]
        logger.info(coef_df.sort_values("COEF"))

        coef_df = coef_df.sort_values("COEF")

        self.rush_mult_df = coef_df
        coef_df.to_csv(r"/Users/jackarmand/Documents/GitHub/vorp-nfl/vorp_nfl/data/rb_vorp_mult.csv")

if __name__ == "__main__":
    from vorp_nfl.preprocessing import extract_player, create_time_features
    import pandas as pd
    from vorp_nfl.log_init import initialize_logger

    initialize_logger()

    df = pd.read_csv("/Users/jackarmand/Documents/GitHub/vorp-nfl/vorp_nfl/data/pbp-2021.csv")
    create_time_features(df)
    extract_player(df)

    vorp = VORP(df, continuous_features=["ToGo", "YardLine", "DateInt", "GameTime"])

    params = {
        "learning_rate":1e-2,
        "n_estimators":1000,
        "num_leaves":15,
        "max_depth":10,
        "reg_lambda":1e3
    }

    vorp.train_rush_model(params)
    vorp.rush_player_basic_vorp()
    vorp.rush_player_multiplier_vorp()