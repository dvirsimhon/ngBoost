import RandomForestRegressor
import ngBoost

def main():
    # RandomForestRegressor.optimize_hyper_parameters_dt()
    RandomForestRegressor.rfr_comparision()
    # ngBoost.optimize_hyper_parameters_dt()
    ngBoost.ngBoost_comparision()
if __name__ == "__main__":
    main()
