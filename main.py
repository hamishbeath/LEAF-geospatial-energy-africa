import numpy as np
import pandas as pd
from Utils import *


class LEAF:

    model_filepath = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF/'
    model_inputs, model_outputs = model_filepath + 'Inputs/', model_filepath + 'Outputs/'
    sensitivity_outputs = model_outputs + 'Sensitivity/'
    sensitivity_inputs = model_inputs + 'Sensitivity/'
    country_data_sheet = pd.read_csv(model_inputs + 'country_inputs.csv', index_col=None)
    country_energy_subs = pd.read_csv(model_inputs + 'energy/energy_file_inputs.csv')
    grid_emissions, household_size = country_data_sheet['Grid Emissions Intensity (kgCO2/kWh)'], \
                                     country_data_sheet['Household Size']
    # ssps = ['2']
    ssp = 2
    years = ['2020']
    CLOVER_load_filepaths = '/Volumes/Hamish_ext/Mitigation_project/CLOVER_inputs/Load/Load_by_year/'
    # CLOVER_results_pv = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA'
    #                                 '/Results/SSP2_MG_tier2_2020_PV.csv')
    # CLOVER_results_diesel = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA'
    #                                     '/Results/SSP2_MG_tier2_2020_diesel.csv')
    fishnet_filepath = ('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA/SPATIAL/fishnet/')
    target_year = 2030
    scenario = 'universal'  # Universal, Baseline or All
    grid_cost = 16000  # Current USD / km of lines
    grid_connection_cost = 90  # Current USD / household
    install_cost = 250  # per household, applied uniformly across all modes
    extension_emissions_factor = 3000
    current_calendar_year = 2022  # Used for discounting calculation
    reliability = 'full'  # 'full or 'grid'
    discount_rate = 0.08
    off_grid_system = 'MG'
    SHS_density_threshold = 2000
    global_carbon_tax = 'N'  # set to Y to include and N for no - include global carbon tax
    split_emissions_assets = 'Y'  # set to Y to include and N for no - split emissions over asset lifetime
    grid_distribution_network_asset_lifetime = 20
    PV_infrastructure_asset_lifetime = 20
    # global_carbon_taxes = {2020: 123, 2025: 123, 2030: 205, 2035: 273}  # Carbon tax scheme in current USD
    global_carbon_taxes = {2020: 3.63, 2025: 87.82, 2030: 145.87, 2035: 200.37}
    # carbon_taxes_2010 = {2020: 2.71, 2025: 65.52, 2030: 108.83, 2035: 200.37}  # Carbon tax scheme in current USD
    unmet_demand_penalty = 0.0  # $/kWh
    met_demand_subsidy = 0.0    # $/kWh
    sensitivity_analysis = 'Y'   # Set to 'Y' or 'N'
    modes = ['pv_mini_grid', 'diesel_mini_grid', 'diesel_SA', 'pv_SA', 'grid']
    include_grid_generation = 'Y' # 'Y' or 'N' for whether to include the additional generating capacity on the grid

"""
Class for spatial dataset variables
"""


class SpatialData:
    spatial_filepath = LEAF.model_filepath + 'Spatial/'

    # File containing the historical calculation of trend access, and baseline
    no_access_historical = pd.read_csv(spatial_filepath + 'No_access_pop/baselines_historical_cal.csv')
    # population_projections = pd.read_csv(spatial_filepath + 'pop_projection_all.csv')
    # population_by_year = pd.read_csv(spatial_filepath + 'population_by_year_rm_d.csv')
    no_access_2020 = no_access_historical['access_2020']
    pop_2020 = no_access_historical['pop_2020']
    no_access_annual_change = pd.read_csv(spatial_filepath + 'no_access_change_by_year.csv')
    calc_sheet = pd.read_csv(spatial_filepath + 'No_access_pop/sheet_for_calc.csv')

    # Grid
    existing_grid_distance_to = pd.read_csv(spatial_filepath + 'existing_grid_length.csv')
    existing_grid_density = pd.read_csv(spatial_filepath + 'existing_grid_density.csv')


def main() -> None:

    # population_access_per_year(LEAF.target_year)

    # annual_population_added_output(LEAF.target_year, LEAF.scenario)
    # return_households_added_by_cell(LEAF.scenario, LEAF.target_year)
    # for i in range(1, 5):

        # electricity_demand_by_year(LEAF.target_year, LEAF.scenario, i)
        # calculate_emissions_investment_national_grid(LEAF.target_year, LEAF.scenario, i)
        # calculate_emissions_investment_off_grid(LEAF.target_year, LEAF.scenario, i, LEAF.reliability, 'MG')
        # calculate_emissions_investment_off_grid(LEAF.target_year, LEAF.scenario, i, LEAF.reliability, 'SHS')
        # calculate_emissions_investment_off_grid(LEAF.target_year, LEAF.scenario, i, 'grid', 'MG')
        # calculate_emissions_investment_off_grid(LEAF.target_year, LEAF.scenario, i, 'grid', 'SHS')

    # for i in range(3, 5):
    #     return_least_cost_option(LEAF.target_year, LEAF.scenario, i, LEAF.reliability, LEAF.unmet_demand_penalty,
    #                              LEAF.global_carbon_taxes, LEAF.met_demand_subsidy)
    reliability_subsidy_sensitivity_analysis()
    # Utils().calculate_grid_coverage_by_country()
    # Utils().scenario_analysis(4, 'CT_Y_RP0.0', 'all')
    # input_file = pd.read_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF/Spatial/grid_extension_emissions_universal_2030_tier_3.csv')
    # Utils().adapt_infrastructure_emissions(input_file, LEAF.target_year, 20)
    # Utils().grid_cost_and_cf()
    # Utils().categorise_countries_by_demand_growth(2030)
    # reliability_penalty_sensitivity_analysis()
    # carbon_sensitivity_analysis()
    # Utils().household_installation_cost(LEAF.target_year)

def reliability_penalty_sensitivity_analysis():
    """
    Perform sensitivity analysis on the reliability penalty.

    This function iterates over a range of reliability penalties and calculates the least cost option
    for each penalty. It saves the results to CSV files.

    Parameters:
    None

    Returns:
    None
    """

    # Iterate over different tiers
    for i in range(2, 5):
        tier_results_output = pd.DataFrame()
        reliability_penalties = np.arange(0.0, 1.05, 0.05)
        
        # Iterate over different reliability penalties
        for penalty in reliability_penalties:
            unmet_demand_penalty = penalty
            
            # Calculate least cost option for the given parameters
            tier_penalty_output, tier_penalty_by_cell = \
                return_least_cost_option(LEAF.target_year, LEAF.scenario, i, LEAF.reliability, unmet_demand_penalty,
                                         LEAF.global_carbon_taxes)
            
            # Set the index of the output DataFrame to the penalty value
            tier_penalty_output.index = [penalty]
            
            # Concatenate the output DataFrame with the overall results DataFrame
            tier_results_output = pd.concat([tier_results_output, tier_penalty_output], axis=0)
            
            # Save the results by cell to a CSV file
            tier_penalty_by_cell.to_csv(LEAF.sensitivity_outputs + 'Reliability_penalty_sensitivity_analysis_'
                                        + str(penalty) +'_by_cell_tier_' + str(i) + '.csv')
        
        # Save the overall results to a CSV file
        tier_results_output.to_csv(LEAF.sensitivity_outputs
                                   + 'Reliability_penalty_sensitivity_analysis_tier' + str(i) + '.csv')

# Function to perform sensitivity analysis on reliability subsidies
def reliability_subsidy_sensitivity_analysis():
    """
    Perform sensitivity analysis on reliability subsidies for different tiers.
    
    This function iterates over different tiers and different reliability subsidies to calculate the least cost option
    for the given parameters. It saves the results to CSV files.
    """
    # Iterate over different tiers
    for i in range(2, 5):
        tier_results_output = pd.DataFrame()
        reliability_subsidies = np.arange(0.0, 1.05, 0.05)
        
        # Iterate over different reliability subsidies
        for subsidy in reliability_subsidies:
            
            # Calculate least cost option for the given parameters
            tier_subsidy_output, tier_subsidy_by_cell = \
                return_least_cost_option(LEAF.target_year, LEAF.scenario, i, LEAF.reliability, 0,
                                         LEAF.global_carbon_taxes, subsidy)
            
            # Set the index of the output DataFrame to the subsidy value
            tier_subsidy_output.index = [subsidy]
            
            # Concatenate the output DataFrame with the overall results DataFrame
            tier_results_output = pd.concat([tier_results_output, tier_subsidy_output], axis=0)
            
            # Save the results by cell to a CSV file
            tier_subsidy_by_cell.to_csv(LEAF.sensitivity_outputs + 'Reliability_subsidy_sensitivity_analysis_'
                                        + str(subsidy) +'_by_cell_tier_' + str(i) + '.csv')
        
        # Save the overall results to a CSV file
        tier_results_output.to_csv(LEAF.sensitivity_outputs
                                   + 'Reliability_subsidy_sensitivity_analysis_tier' + str(i) + '.csv')

def carbon_sensitivity_analysis():

    if LEAF.global_carbon_tax == 'N':
        print('Carbon tax not set to be included in the analysis')
        return

    # Import carbon prices
    carbon_prices = pd.read_csv(LEAF.sensitivity_inputs + 'carbon_prices.csv', index_col=0)
    for i in range(2, 5):
        tier_results_output = pd.DataFrame()
        for decile in carbon_prices.index:
            row_select = carbon_prices.loc[decile]
            current_year = 2020
            decile_carbon_prices = {}
            for year in range(0, 4):
                year_tax = row_select[str(current_year)]
                decile_carbon_prices[current_year] = year_tax
                current_year += 5
            carbon_price_output, tier_results_by_cell  \
                = return_least_cost_option(LEAF.target_year, LEAF.scenario, i, LEAF.reliability,
                                                           LEAF.unmet_demand_penalty, decile_carbon_prices)
            carbon_price_output.index = [decile]
            tier_results_output = pd.concat([tier_results_output, carbon_price_output], axis=0)
            tier_results_by_cell.to_csv(LEAF.sensitivity_outputs + 'Carbon_price_sensitivity_analysis_'
                                        + str(decile) +'_by_cell_tier_' + str(i) + '.csv')
        # tier_results_output.to_csv(LEAF.sensitivity_outputs
        #                            + 'Carbon_sensitivity_analysis_tier' + str(i) + '.csv')

def population_access_per_year(target_year):
    """
    Function that returns the annual population gaining access and spatial distribution of this, either for the baseline
    scenarios or the universal one.

    Inputs:
        - No Access Levels in base year (by cell)
        - Calculated baseline trend change based on historical data (by cell)
        - Share of population growth (by cell) that falls into 'no access'
        - Expected population growth by year (by cell)
    Outputs:
        - By-cell and by-country outputs giving the baseline estimates and the universal estimates up to the target year
    """

    # Import and set variables used for function
    countries = LEAF.country_data_sheet['ISO3']

    # Pull in base population, share, no access in base, growth rates, annual baseline change
    no_access_pop_change = SpatialData.no_access_annual_change  # The annual change in no access population by cell
    baseline_change = SpatialData.no_access_historical['Annual Change']

    # Create Dataframes which will be appended
    baseline_no_access_by_year = pd.DataFrame([])
    baseline_change_by_year = pd.DataFrame([])
    baseline_summary = pd.DataFrame([])
    #
    # # Set 2020 value in by-year access
    baseline_no_access_by_year['2020'] = SpatialData.no_access_historical['access_2020']
    #
    # Set country list values in summary sheet
    baseline_summary['ISO3'], baseline_summary['Country'] = \
        LEAF.country_data_sheet['ISO3'], LEAF.country_data_sheet['location']
    #
    # # Calculate years assessed variable
    years_assessed = target_year - 2020

    baseline_electrified = {}

    # Loop through years up to target year
    for year in range(2021, target_year + 1):

        print('Calculating Baseline Access Amounts for', str(year))

        # Set up frame to produce outputs with
        calc_sheet = pd.DataFrame([])
        calc_sheet['ISO3'] = SpatialData.calc_sheet['ISO3']
        calc_sheet['id'] = SpatialData.calc_sheet['id']

        # Target year population change by cell - respective column
        year_no_access_pop_change = no_access_pop_change[str(year)]  # Variable set for current year change by cell

        # Calculate the cell values by cell before pre-adjustment
        year_value_pre_adjustment = baseline_no_access_by_year[str(year - 1)] \
                                    + year_no_access_pop_change + baseline_change
        calc_sheet['pre_adjustment'] = year_value_pre_adjustment
        calc_sheet_indexed = calc_sheet.set_index('ISO3')
        calc_sheet_indexed = calc_sheet_indexed.fillna(0)

        # Empty Lists for outputs
        fid, values = [], []  # ids, values lists

        # Loop through each country
        for country in countries:
            # Make a country dictionary within baseline_electrified if not created already
            try:
                test = baseline_electrified[country]
            except KeyError:
                baseline_electrified[country] = {}
            # Pull out selected cells by country
            country_cells = calc_sheet_indexed.loc[country]

            negative_country_cells = country_cells.loc[country_cells['pre_adjustment'] < 0]
            positive_country_cells = country_cells.loc[country_cells['pre_adjustment'] >= 0]
            remaining_balance_trend_access = (np.sum(negative_country_cells['pre_adjustment'])) * - 1

            # Append fid and zero values of the cells that were made negative
            for cell in range(0, len(negative_country_cells)):
                fid.append(negative_country_cells['id'][cell])
                values.append(0)

            # Arrange cells with positive balance of population
            positive_country_cells_ascending = positive_country_cells.sort_values(by='pre_adjustment', axis=0,
                                                                                  ascending=False)
            remaining_balance_needing_access = (np.sum(positive_country_cells_ascending['pre_adjustment']))
            it, adding_pop = 0, 0

            # Check to see if universal access in country predicted to be met, if so append values to zero.
            if remaining_balance_needing_access < remaining_balance_trend_access:
                for cell in range(0, len(positive_country_cells)):
                    fid.append(positive_country_cells['id'][cell])
                    values.append(0)

            # If universal access not expected to be met, fill in by cells with the largest population first
            else:
                # While loop, that continues to add cells as the total remaining trend is less than the amount added
                while adding_pop < remaining_balance_trend_access:
                    fid.append(positive_country_cells_ascending['id'][it])
                    values.append(0)
                    try:
                        baseline_electrified[country].update({positive_country_cells_ascending['id'][it]: year})
                    except:
                        baseline_electrified[country][positive_country_cells_ascending['id'][it]] = year
                    adding_pop += positive_country_cells_ascending['pre_adjustment'][it]
                    it += 1

                # Loop that leaves the amounts of population calculated for the year in place as reached target level
                for remaining_cells in range(it, len(positive_country_cells_ascending)):
                    values.append(
                        positive_country_cells_ascending['pre_adjustment'][remaining_cells])
                    fid.append(positive_country_cells_ascending['id'][remaining_cells])

        # Prepare data for appending to df, arranging by ascending id to match other outputs
        to_add = list(zip(fid, values))
        to_add = pd.DataFrame(to_add, columns=['id', 'values'])
        to_add_sorted = to_add.sort_values(by='id', axis=0, ascending=True)
        to_add_sorted = to_add_sorted.reset_index(drop=True)
        baseline_no_access_by_year[str(year)] = to_add_sorted['values']

        # Append the baseline year change value to df
        baseline_change_by_year[str(year)] = \
            baseline_no_access_by_year[str(year)] - baseline_no_access_by_year[str(year - 1)]

    # Spit out the baseline dfs to CSV, the dfs are used in the following part of the function
    baseline_no_access_by_year_output = pd.concat([SpatialData.calc_sheet, baseline_no_access_by_year], axis=1)
    baseline_change_by_year_output = pd.concat([SpatialData.calc_sheet, baseline_change_by_year], axis=1)

    baseline_no_access_by_year_output.to_csv(SpatialData.spatial_filepath +
                                             'No_access_pop/baseline_annual_' + str(target_year) + '.csv')
    baseline_change_by_year_output.to_csv(SpatialData.spatial_filepath +
                                          'No_access_pop/baseline_annual_change_' + str(target_year) + '.csv')

    # Create summary sheet
    baseline_summary_sheet = pd.DataFrame([])
    baseline_summary_sheet['ISO3'] = countries
    baseline_no_access_by_year_indexed = baseline_no_access_by_year_output.set_index('ISO3')
    for summary_year in range(2020, target_year + 1):
        year_totals = []
        for summary_country in countries:
            summary_cells = baseline_no_access_by_year_indexed.loc[summary_country]
            year_totals.append(np.sum(summary_cells[str(summary_year)]))
        baseline_summary_sheet[str(summary_year)] = year_totals

    baseline_summary_sheet.to_csv(SpatialData.spatial_filepath +
                                  'No_access_pop/baseline_summary_' + str(target_year) + '.csv')

    # Calculation of meeting universal access in the years set by the target
    # baseline_summary = baseline_summary_sheet
    # baseline_no_access_by_year = baseline_no_access_by_year_output
    baseline_summary = \
        pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/baseline_summary_' + str(target_year) + '.csv')
    baseline_no_access_by_year = pd.read_csv(SpatialData.spatial_filepath +
                                             'No_access_pop/baseline_annual_' + str(target_year) + '.csv')
    baseline_no_access_by_year_output = baseline_no_access_by_year
    no_access_by_year = pd.DataFrame([])
    change_by_year = pd.DataFrame([])
    summary = pd.DataFrame([])

    # # Set 2020 value in by-year access
    # no_access_by_year['2020'] = SpatialData.no_access_historical['access_2020']

    # Set country list values in summary sheet
    summary['ISO3'], summary['Country'] = LEAF.country_data_sheet['ISO3'], LEAF.country_data_sheet['location']

    # Set index for the by year, by cell, baseline amounts without access.
    baseline_no_access_by_year_iso_index = baseline_no_access_by_year.set_index('ISO3')
    baseline_no_access_by_year_iso_index.to_csv(SpatialData.spatial_filepath + 'baseline_no_access_iso_index.csv')
    total_neg = 0
    # Work through countries to calculate their expected path to universal access
    for country in countries:

        print('Running the universal access calculation for', country)
        country_cells = baseline_no_access_by_year_iso_index.loc[country]
        country_cells_dropped = country_cells.drop(['Unnamed: 0', '2020'], axis=1)  # Remove additional columns

        # Split the cells between those that need additional investment above baseline and those that don't
        target_country_cells = country_cells_dropped.loc[country_cells[str(target_year)] > 0]
        non_processed_cells = country_cells_dropped.loc[country_cells[str(target_year)] <= 0]

        # Deal with the baseline electrified cells for the given country
        country_baseline_electrified = baseline_electrified[country]
        country_baseline_electrified_popped = dict(country_baseline_electrified)
        non_processed_checker = pd.DataFrame(non_processed_cells.set_index('id'))

        # Remove any of the baseline electrified cells that reach universal electrification
        for (key, value) in country_baseline_electrified.items():
            if key in non_processed_checker.index:
                del country_baseline_electrified_popped[key]
            else:
                pass

        # Append the cells that are projected to reach universal access
        no_access_by_year = pd.concat([no_access_by_year, non_processed_cells], axis=0)

        # Skip if respective country reaches universal electrification in all cells by target year
        if len(target_country_cells) == 0:
            pass

        # Else run the process of reaching universal access
        else:

            # Create empty list of cells that have been electrified
            electrified_cells = []
            country_out_df = pd.DataFrame([])
            country_net_changes = pd.DataFrame([])
            cumulative_cell_change = np.array(np.zeros(len(target_country_cells)))

            # Work through the years
            for year_universal in range(2021, target_year + 1):
                print(year_universal)
                integer_year = year_universal - 2020
                # Extract year cells
                year_country_cells = pd.DataFrame([])
                year_country_cells[str(year_universal)] = target_country_cells[str(year_universal)]
                year_country_cells['id'] = target_country_cells['id']

                # If not start year, then factor in the previous year net change
                if year_universal > 2021:
                    # Subtracts the cumulative net gains from the year country cells so that its only 'new' population
                    year_country_cells_ls = year_country_cells[str(year_universal)].tolist()
                    print('The cumulative cell change for', year_universal, 'is:', np.sum(cumulative_cell_change))
                    year_country_cells_ls = np.add(year_country_cells_ls, cumulative_cell_change)
                    year_country_cells_df = pd.DataFrame([year_country_cells_ls.tolist(), year_country_cells['id']]).T
                    # year_country_cells_df.to_csv(SpatialData.spatial_filepath + 'test11.csv')
                    year_country_cells_df.columns = [str(year_universal), 'id']
                    year_country_cells_df_id_index = year_country_cells_df.set_index('id')
                    year_country_cells = year_country_cells_df_id_index.clip(lower=0)
                    # if year_universal > 2022:
                    #     break
                # Calculate target for given year
                # year_target = np.sum(year_country_cells[str(year_universal)]) * (integer_year / years_assessed)

                power_bias = 2
                year_target = np.sum(year_country_cells[str(year_universal)]) * ((integer_year / years_assessed)
                                                                                 ** power_bias)
                # year_target = np.sum(year_country_cells[str(year_universal)])
                print('Year target:', year_target)
                # Arrange cells in order of the largest first, this is to meet the denser cells first as electrification
                # strategy.
                year_country_cells_asc_df = \
                    year_country_cells.sort_values(axis=0, by=str(year_universal), ascending=False)

                year_country_cells_asc_df = year_country_cells_asc_df.reset_index()
                # year_country_cells_asc_df.to_csv(SpatialData.spatial_filepath + 'test_02.csv')

                # Current year lists keeping track of
                new_values, net_change, fids = [], [], []

                # Variable to keep track of the population given access in a given year
                given_access = 0

                # Add the baseline electrified cells for given year to the electrified cells list, to add trend growth
                for (key, value) in country_baseline_electrified_popped.items():
                    if value == year_universal:
                        electrified_cells.append(key)

                # If cells have already been connected, ensure that the trend change is met.
                if len(electrified_cells) > 0:

                    # Separate out already electrified cells
                    electrified_cells_df = \
                        year_country_cells_asc_df[year_country_cells_asc_df['id'].isin(electrified_cells)]

                    # Separate out non-electrified cells
                    year_country_cells_asc_df = \
                        year_country_cells_asc_df[~year_country_cells_asc_df['id'].isin(electrified_cells)]

                    # Reset indexes
                    electrified_cells_df = electrified_cells_df.reset_index(drop=True)
                    year_country_cells_asc_df = year_country_cells_asc_df.reset_index(drop=True)
                    year_country_net_amount = 0

                    # Run through electrified cells and append values accordingly
                    for electrified in range(0, len(electrified_cells_df)):
                        new_values.append(0)
                        net_change.append(-electrified_cells_df[str(year_universal)][electrified])
                        fids.append(electrified_cells_df['id'][electrified])
                        given_access += electrified_cells_df[str(year_universal)][electrified]
                        year_country_net_amount += electrified_cells_df[str(year_universal)][electrified]

                    print('Given access after electrified:', given_access)
                # Work through cells and connect by the most dense first, up to the target year value
                it = 0
                while given_access < year_target and it < len(year_country_cells_asc_df):
                    if year_country_cells_asc_df['id'][it] in country_baseline_electrified_popped.keys():
                        new_values.append(
                            year_country_cells_asc_df[str(year_universal)][it])
                        net_change.append(0)
                        fids.append(year_country_cells_asc_df['id'][it])
                    else:
                        given_access += year_country_cells_asc_df[str(year_universal)][it]  # Add cell pop to given access
                        new_values.append(0)  # Append new values to zero
                        net_change.append(
                            -year_country_cells_asc_df[str(year_universal)][it])  # Append the net change values
                        fids.append(year_country_cells_asc_df['id'][it])  # Append cell id to list
                        electrified_cells.append(year_country_cells_asc_df['id'][it])  # Mark cell as 'electrified'
                    it += 1  # Add one to iteration count

                # Now work through the cells that don't get dealt with or 'electrified' this time
                for remaining in range(it, len(year_country_cells_asc_df)):
                    new_values.append(
                        year_country_cells_asc_df[str(year_universal)][remaining])  # Append new values to the same
                    net_change.append(0)  # Append net change as zero
                    fids.append(year_country_cells_asc_df['id'][remaining])  # Append cell id to list

                # Reorder and compile things to be appended into the required outputs
                to_reorder = list(zip(new_values, fids, net_change))
                year_frame = pd.DataFrame(to_reorder, columns=[str(year_universal), 'id', 'Net_change'])
                year_frame = year_frame.sort_values(by='id', axis=0, ascending=True)
                year_frame.reset_index(drop=True, inplace=True)
                year_frame.to_csv(SpatialData.spatial_filepath + 'test12.csv')
                country_out_df[str(year_universal)] = year_frame[str(year_universal)]
                # country_out_df.to_csv(SpatialData.spatial_filepath + 'test05.csv')
                country_net_changes[str(year_universal)] = year_frame['Net_change']
                # cumulative_cell_change_df1 = pd.DataFrame(cumulative_cell_change)
                # cumulative_cell_change_df1.to_csv(SpatialData.spatial_filepath + 'test06.csv')
                cumulative_cell_change = np.add(cumulative_cell_change, year_frame['Net_change'].values)
                # cumulative_cell_change_df2 = pd.DataFrame(cumulative_cell_change)
                # cumulative_cell_change_df2.to_csv(SpatialData.spatial_filepath + 'test07.csv')
                # print(cumulative_cell_change_df2)
                # if year_universal > 2021:
                #     break
            # Export at the country level after processing the year data
            # target_country_cells = target_country_cells.reset_index()

            target_country_cells = target_country_cells.reset_index().set_index('id')
            country_out_df['id'] = target_country_cells.index
            country_out_df = country_out_df.set_index('id')
            country_out_df['ISO3'] = target_country_cells['ISO3']
            country_out_df['ADM0_NAME'] = target_country_cells['ADM0_NAME']
            country_out_df = country_out_df.reset_index().set_index('ISO3')
            no_access_by_year = pd.concat([no_access_by_year, country_out_df], axis=0)

    # Deal with the final output sheet and add various things in.
    no_access_by_year = no_access_by_year.sort_values(by='id', axis=0, ascending=True)

    # Create universal access summary sheet
    print('Creating summary outputs')
    universal_summary_sheet = pd.DataFrame([])
    universal_summary_sheet['ISO3'] = countries
    for summary_year in range(2021, target_year + 1):
        year_totals = []
        for summary_country in countries:
            summary_cells = no_access_by_year.loc[summary_country]
            year_totals.append(np.sum(summary_cells[str(summary_year)]))
        universal_summary_sheet[str(summary_year)] = year_totals

    universal_summary_sheet.to_csv(SpatialData.spatial_filepath +
                                   'No_access_pop/universal_summary_' + str(target_year) + '.csv')
    print(baseline_no_access_by_year)
    no_access_by_year = no_access_by_year.reset_index()
    no_access_by_year['2020'] = baseline_no_access_by_year['2020']
    no_access_by_year = no_access_by_year.drop(['Unnamed: 0.1'], axis=1)
    no_access_by_year.to_csv(SpatialData.spatial_filepath + 'No_access_pop/universal_annual_'
                             + str(target_year) + '.csv', index=False)

def return_households_added_by_cell(scenario, target_year):
    """
    Function that returns the households added by cell, based on households size in each country

    Inputs:
        - Country-level data on household size
        - The population added in each cell up to target year for given scenario

    :param scenario: the scenario being run, baseline, universal or all
    :param target_year: year by which universal access is reached
    :return: (additional) households in each cell gaining access each year

    """
    # Import necessary files and variables
    population_added_by_year = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/by_year_additions_'
                                           + scenario + '_' + str(target_year) + '.csv')
    households_by_country, countries = LEAF.country_data_sheet['Household Size'], LEAF.country_data_sheet['ISO3']
    population_added_by_year_iso_index = population_added_by_year.set_index('ISO3')

    # Create output frame
    output_df = pd.DataFrame([])

    # Work through each country by ISO3 code
    for country in range(0, len(countries)):
        country_cells = population_added_by_year_iso_index.loc[countries[country]]
        country_household_size = households_by_country[country]
        country_cells_clipped = country_cells.loc[:, '2021': str(target_year)]
        country_cells_households = country_cells_clipped / country_household_size
        country_cells_households = pd.DataFrame(country_cells_households)
        country_cells_households = pd.concat([country_cells['id'], country_cells_households], axis=1)
        output_df = pd.concat([output_df, country_cells_households], axis=0)

    output_df = output_df.sort_values(by='id', axis=0, ascending=True)
    output_df = output_df.reset_index()
    output_df = pd.concat([population_added_by_year['ADM0_NAME'], output_df], axis=1)
    output_df['sum'] = output_df.loc[:, '2021': str(target_year)].sum(axis=1)
    output_df.to_csv(SpatialData.spatial_filepath + 'households_added_by_year_' + scenario + '_' +
                     str(target_year) + '.csv')

def annual_population_added_output(target_year, scenario):
    """
    Function that returns the annual population added at cell-level for universal access scenario

    Inputs:
        - Baseline and universal by-cell and by-year data
        - Target year (should align with the above)
    Outputs:
        - Additional population to be added by-year at the cell level for the universal scenario, to be used in
        function looking at equivalent households.
    To do:
        - add in scenario option for baseline, universal or all and do respective outputs for that.

    """
    # Import necessary baseline and universal files
    baseline_annual = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/baseline_annual_'
                                  + str(target_year) + '.csv')
    universal_annual = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/universal_annual_'
                                   + str(target_year) + '.csv')

    # Create empty initial df to subtract one from the other
    subtracted_df = pd.DataFrame([])

    # Minus the universal from the baseline amounts
    for year in range(2021, target_year + 1):
        subtracted_df[str(year)] = baseline_annual[str(year)] - universal_annual[str(year)]

    output_df = pd.DataFrame([])
    # Add columns to output datframe to be consistent
    output_df['ISO3'], output_df['ADM0_NAME'], output_df['id'] = universal_annual['ISO3'], \
                                                                 universal_annual['ADM0_NAME'], universal_annual['id']
    output_df['2021'] = subtracted_df['2021']
    output_df['sum'] = np.zeros(len(universal_annual))

    # Loop through years and calculate the additional added each year in each cell
    for year in range(2022, target_year + 1):
        print('Preparing outputs for', str(year))
        year_column = []
        for row in range(0, len(universal_annual)):
            if subtracted_df[str(year)][row] > subtracted_df[str(year - 1)][row]:
                year_column.append(subtracted_df[str(year)][row] - subtracted_df[str(year - 1)][row])
            else:
                year_column.append(0)
        output_df[str(year)] = year_column
        output_df['sum'] += year_column

    output_df['sum'] += output_df['2021']
    # Output to CSV
    output_df.to_csv(SpatialData.spatial_filepath + 'No_access_pop/by_year_additions_' + scenario + '_' +
                     str(target_year) + '.csv')

def annual_population_added_output_old(target_year, scenario):
    """
    Function that returns the annual population added at cell-level for universal access scenario

    Inputs:
        - Baseline and universal by-cell and by-year data
        - Target year (should align with the above)
    Outputs:
        - Additional population to be added by-year at the cell level for the universal scenario, to be used in
        function looking at equivalent households.
    To do:
        - add in scenario option for baseline, universal or all and do respective outputs for that.

    """
    # Import necessary baseline and universal files
    baseline_annual = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/baseline_annual_'
                                  + str(target_year) + '.csv')
    universal_annual = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/universal_annual_'
                                   + str(target_year) + '.csv')

    # Create empty initial df to subtract one from the other
    subtracted_df = pd.DataFrame([])

    # Minus the universal from the baseline amounts
    for year in range(2021, target_year + 1):
        subtracted_df[str(year)] = baseline_annual[str(year)] - universal_annual[str(year)]

    output_df = pd.DataFrame([])
    # Add columns to output datframe to be consistent
    output_df['ISO3'], output_df['ADM0_NAME'], output_df['id'] = universal_annual['ISO3'], \
                                                                 universal_annual['ADM0_NAME'], universal_annual['id']
    output_df['2021'] = subtracted_df['2021']
    output_df['sum'] = np.zeros(len(universal_annual))

    # Loop through years and calculate the additional added each year in each cell
    for year in range(2022, target_year + 1):
        print('Preparing outputs for', str(year))
        year_column = []
        for row in range(0, len(universal_annual)):
            if subtracted_df[str(year)][row] > subtracted_df[str(year - 1)][row]:
                year_column.append(subtracted_df[str(year)][row] - subtracted_df[str(year - 1)][row])
            else:
                year_column.append(0)
        output_df[str(year)] = year_column
        output_df['sum'] += year_column

    # Output to CSV
    output_df.to_csv(SpatialData.spatial_filepath + 'No_access_pop/by_year_additions_' + scenario + '_' +
                     str(target_year) + '.csv')

def electricity_demand_by_year(target_year, scenario, tier_of_access):
    """
    Function that calculates the energy demand in each cell over time.

    :param target_year:
    :param scenario:
    :param tier_of_access:
    :return:

    """

    # Import required files to run the analysis
    households_added_by_year = pd.read_csv(SpatialData.spatial_filepath + 'households_added_by_year_'
                                           + scenario + '_' + str(target_year) + '.csv')
    households_added_by_year_indexed = households_added_by_year.set_index('ISO3')

    countries = LEAF.country_data_sheet['ISO3']
    energy_subs = LEAF.country_energy_subs['energy_file']

    output_df = pd.DataFrame([])

    # Loop through each country
    for country in range(0, len(countries)):

        # Get country full name (in some cases with substitute used due to lack of data)
        lookup_country = energy_subs[country]

        # Create dictionary containing household level year totals for the current country
        household_sum_dict = {}
        for year in range(2021, target_year + 1):
            # energy_file = (pd.read_csv(LEAF.CLOVER_load_filepaths + 'SSP' + str(LEAF.ssp) + '/Tier '
            #                            + str(tier_of_access) + '/' + str(lookup_country)
            #                            + '_community_load_' + str(year) + '.csv')[str(year)]) / 100
            energy_file = \
                (pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/CLOVER DEMAND BACKUP/'
                             + 'SSP' + str(LEAF.ssp) + '/Tier ' + str(tier_of_access) + '/' + str(lookup_country) +
                             '_community_load_' + str(year) + '.csv')[str(year)]) / 100

            year_energy_total = np.sum(energy_file) / 1000
            household_sum_dict[year] = year_energy_total

        # Select relevant country and year cells to create energy demand totals
        country_cells = households_added_by_year_indexed.loc[countries[country]]

        # Create output df to add analysed cells to.
        energy_demand_output_df = pd.DataFrame(columns=range(2021, target_year + 1), index=country_cells.index)
        energy_demand_output_df = energy_demand_output_df.fillna(0)
        energy_demand_output_df['id'] = country_cells['id']

        # Loop through all years and add in trajectory of energy demand up to target year
        for year in range(2021, target_year + 1):

            # Select households electrified in
            year_households_electrified = country_cells[str(year)]

            # Loop through remaining years
            for remaining_years in range(year, target_year + 1):
                # Add to the respective column to the existing year one in the output dataframe
                energy_demand_output_df[remaining_years] += (year_households_electrified *
                                                             household_sum_dict[remaining_years])

        output_df = pd.concat([output_df, energy_demand_output_df])

    output_df = output_df.sort_values(by='id', axis=0, ascending=True)
    output_df = output_df.reset_index()
    output_df = pd.concat([households_added_by_year['ADM0_NAME'], output_df], axis=1)
    # add 'sum' of all years column
    output_df['sum'] = np.sum(output_df.loc[:,2021:2030], axis=1)
    output_df.to_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_' + scenario + '_' +
                     str(target_year) + '_tier_' + str(tier_of_access) + '.csv')

def calculate_emissions_investment_national_grid(target_year, scenario, tier_of_access):

    # Import files needed
    grid_density = SpatialData.existing_grid_density['LENGTH']
    grid_distance_to = (SpatialData.existing_grid_distance_to['NEAR_DIST']) / 1000
    energy_by_id_year = pd.read_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_' +
                                    scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    households_added = \
        pd.read_csv(SpatialData.spatial_filepath + 'households_added_by_year_' + scenario + '_' +
                    str(target_year) + '.csv')
    # Set variables
    years_examined = target_year - 2021

    # Create grid binary of whether grid present already. If present, zero used.
    grid_binary = []
    for fid in range(0, len(grid_density)):
        if grid_density[fid] > 0:
            grid_binary.append(0)
        else:
            grid_binary.append(1)

    # Output pandas series of construction costs and emissions
    grid_costs = grid_distance_to * LEAF.grid_cost * grid_binary
    grid_construction_emissions = grid_distance_to * grid_binary * LEAF.extension_emissions_factor

    # Get grid capacity costs from util function
    grid_capacity_costs = Utils().grid_capacity_cost(energy_by_id_year, tier_of_access)

    # Create energy sheet with index of ISO3
    energy_fid_iso_indexed = energy_by_id_year.set_index('ISO3')  # Set index of country

    # Set empty dataframe for the emissions resulting from demand to append values below to
    emissions_from_demand_df = pd.DataFrame([])

    # Set empty dataframe for country cells to be appended to for infrastructure build
    infrastructure_out_df = pd.DataFrame([])

    country_info = LEAF.country_data_sheet

    # Loops through countries included in list and calculates the relevant grid emissions and costs
    for country in range(0, len(country_info)):
        print('Running grid calculations for', country_info['location'][country])
        iso_select, country_emissions_intensity = country_info['ISO3'][country], \
                                                  country_info['Grid Emissions Intensity (kgCO2/kWh)'][country]
        rural_energy_met = 1 - (country_info['Blackout Threshold'][country])  # Extract the reliability threshold
        cells_select = pd.DataFrame(energy_fid_iso_indexed.loc[iso_select])  # select country relevant cells
        cells_years_select = pd.DataFrame(cells_select.loc[:, '2021': str(target_year)])  # Select the cells of data

        # Calculate the emissions resulting from demand
        country_emissions_from_demand = (cells_years_select * country_emissions_intensity) * rural_energy_met  # In KG of CO2

        # Add in the id column as index and concat to the main emissions from demand sheet
        country_emissions_from_demand['id'] = cells_select['id']
        country_emissions_from_demand = country_emissions_from_demand.set_index('id')
        emissions_from_demand_df = pd.concat([emissions_from_demand_df, country_emissions_from_demand], axis=0)

        # Use number index for the selected cells for the looping stage to establish the grid extension year
        cells_select_number_index = cells_select.reset_index()

        # Create empty df of zeros that will be edited depending on when the grid extension happens
        infrastructure_build_df = pd.DataFrame(columns=range(2021, target_year + 1),
                                               index=cells_select_number_index.index)
        infrastructure_build_df = infrastructure_build_df.fillna(0)

        # Loop through the country cells to establish year of construction
        for row in range(0, len(cells_select_number_index)):

            current_row = cells_select_number_index.iloc[[row]]  # Select current row of country cells
            grid_check = 0  # This is a check as to whether
            column_count = 2021  # Counts through the columns of the df

            # Loops through to find year when grid line built
            while grid_check == 0 and column_count <= target_year:

                # if the row amount is above zero,
                if current_row[str(column_count)].values > 0:
                    infrastructure_build_df.at[row, column_count] = 1
                    grid_check += 1
                else:
                    pass
                column_count += 1

        # Output binary of grid infrastructure build
        infrastructure_build_df['id'] = cells_select_number_index['id']

        infrastructure_out_df = pd.concat([infrastructure_out_df, infrastructure_build_df], axis=0)

    # Calculate and construct required outputs for outputting

    # Reorder infrastructure binary into ascending by id
    infrastructure_out_df = infrastructure_out_df.sort_values(by='id', axis=0, ascending=True)
    infrastructure_out_df = infrastructure_out_df.reset_index(drop=True)
    emissions_from_demand_df = emissions_from_demand_df.reset_index()
    emissions_from_demand_df = emissions_from_demand_df.sort_values(by='id', axis=0, ascending=True)
    emissions_from_demand_df = emissions_from_demand_df.reset_index(drop=True)
    grid_extension_emissions_df = pd.DataFrame([])
    grid_extension_costs_df = pd.DataFrame([])
    total_emissions_df = pd.DataFrame([])
    grid_capacity_costs_df = pd.DataFrame([])

    # Loop through and sum together the grid extension and grid demand emissions
    for year in range(2021, target_year + 1):

        grid_extension_emissions_df[str(year)] = infrastructure_out_df[year] * grid_construction_emissions.values
        grid_extension_costs_df[str(year)] = infrastructure_out_df[year] * grid_costs.values
        grid_capacity_costs_df[str(year)] = infrastructure_out_df[year] * grid_capacity_costs.values

        if LEAF.include_grid_generation == 'Y':
            grid_extension_costs_df[str(year)] += grid_capacity_costs_df[str(year)]

        # Factor in grid connection cost
        if LEAF.grid_connection_cost > 0:
            connection_fees = households_added[str(year)] * LEAF.grid_connection_cost
            grid_extension_costs_df[str(year)] += connection_fees

        # if in future years, discount the values accordingly
        if year > LEAF.current_calendar_year:
            grid_extension_costs_df[str(year)] = \
                Utils().return_discounted_value(grid_extension_costs_df[str(year)], year)

        total_emissions_df[str(year)] = grid_extension_emissions_df[str(year)] + emissions_from_demand_df[str(year)]

    grid_extension_emissions_df = pd.concat(
        [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
         grid_extension_emissions_df], axis=1)
    total_emissions_df = pd.concat([energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
                                    total_emissions_df], axis=1)
    emissions_from_demand_df = pd.concat([energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'],
                                          emissions_from_demand_df], axis=1)
    grid_extension_costs_df = pd.concat(
        [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
         grid_extension_costs_df], axis=1)

    grid_capacity_costs = pd.concat(
        [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
         grid_capacity_costs], axis=1)

    # If LEAF.split_emissions_assets == Y, then run alternative emissions calculation
    if LEAF.split_emissions_assets == 'Y':
        grid_extension_emissions_df = Utils().adapt_infrastructure_emissions\
            (grid_extension_emissions_df, LEAF.target_year, LEAF.grid_distribution_network_asset_lifetime)

        grid_extension_emissions_df.columns = grid_extension_emissions_df.columns.astype(str)
        total_emissions_df = grid_extension_emissions_df.loc[:, str(2021):] + emissions_from_demand_df.loc[:, str(2021):]
        total_emissions_df = pd.concat(
            [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
             total_emissions_df], axis=1)

    # Add a sum column to give totals
    grid_extension_emissions_df['sum'] = grid_extension_emissions_df.loc[:, str(2021):].sum(axis=1)
    total_emissions_df['sum'] = total_emissions_df.loc[:, str(2021):].sum(axis=1)
    emissions_from_demand_df['sum'] = emissions_from_demand_df.loc[:, str(2021):].sum(axis=1)
    grid_extension_costs_df['sum'] = grid_extension_costs_df.loc[:, str(2021):].sum(axis=1)
    grid_capacity_costs_df['sum'] = grid_capacity_costs_df.loc[:, str(2021):].sum(axis=1)

    # Export files to csv
    grid_extension_emissions_df.to_csv(SpatialData.spatial_filepath + 'grid_extension_emissions_' +
                                       scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    total_emissions_df.to_csv(SpatialData.spatial_filepath + 'total_emissions_grid_' +
                              scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    emissions_from_demand_df.to_csv(SpatialData.spatial_filepath + 'emissions_from_demand_grid_' +
                                    scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    grid_extension_costs_df.to_csv(SpatialData.spatial_filepath + 'grid_extension_costs_' +
                                   scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    if LEAF.include_grid_generation == 'Y':
        grid_capacity_costs.to_csv(SpatialData.spatial_filepath + 'grid_capacity_costs_' +
                                   scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')

def calculate_emissions_investment_off_grid(target_year, scenario, tier_of_access, reliability, system):
    """
    Function that calculates the  investment requirements and emissions outcomes for electrification via off-grid
    systems

    :param system:
    :param target_year: the target year for universal access
    :param scenario: whether meeting 'baseline', 'universal' or 'all' getting access
    :param tier_of_access: the tier of electricity access
    :param reliability: level of off-grid reliability considered
    :return: returns respective emissions and investments

    """

    # Import energy by cell and households by cell data
    energy_by_id_year = pd.read_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_' + scenario + '_' +
                                    str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    households_added_by_year = pd.read_csv(SpatialData.spatial_filepath + 'households_added_by_year_' +
                                           scenario + '_' + str(target_year) + '.csv')

    # Set variables
    years_examined = target_year - 2021
    # system = 'MG'

    # Pull in country info
    country_info = LEAF.country_data_sheet
    energy_info = LEAF.country_energy_subs
    clover_subs = energy_info['clover_sub']

    # Dictionary to append to
    closest_years = {}

    # Look for database file. If exists, extract. If it doesn't exist run util to build it and return
    try:
        pv_full_reliability = pd.read_csv(Utils.clover_results_filepath + 'pv_full_reliability_SSP_'
                                          + str(LEAF.ssp) + '_tier_' + str(tier_of_access) + '_' + system + '.csv')
        diesel_full_reliability = pd.read_csv(Utils.clover_results_filepath + 'diesel_full_reliability_SSP_'
                                              + str(LEAF.ssp) + '_tier_' + str(tier_of_access) + '_' + system + '.csv')
        pv_grid_reliability = pd.read_csv(Utils.clover_results_filepath + 'pv_grid_reliability_SSP_'
                                          + str(LEAF.ssp) + '_tier_' + str(tier_of_access) + '_' + system + '.csv')
        diesel_grid_reliability = pd.read_csv(Utils.clover_results_filepath + 'diesel_grid_reliability_SSP_'
                                              + str(LEAF.ssp) + '_tier_' + str(tier_of_access) + '_' + system + '.csv')
    except FileNotFoundError:
        # run util function to build databases of relevant results
        pv_full_reliability, diesel_full_reliability, pv_grid_reliability, diesel_grid_reliability = \
            Utils().return_relevant_clover_results_summary(ssp=LEAF.ssp, tier=tier_of_access, system=system)

    if reliability == 'full':
        pv_optimisations = pv_full_reliability
        diesel_optimisations = diesel_full_reliability
    elif reliability == 'grid':
        pv_optimisations = pv_grid_reliability
        diesel_optimisations = diesel_grid_reliability
    else:
        print('LEAF.reliability needs to be set on either "full" or "grid"')

    # set empty dfs for appending
    pv_investment_out, pv_emissions_out, diesel_investment_out, diesel_emissions_out = \
        pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]),

    # Loops through countries included in list and isolates the respective cells (households) and energy demand
    for country in range(0, len(country_info)):
        print('Running mini-grid calculations for', country_info['location'][country])
        country_target_cells = households_added_by_year[households_added_by_year['ISO3']
                                                        == country_info['ISO3'][country]]
        country_target_cells_number_index = country_target_cells.reset_index(drop=True)
        country_clover_sub = clover_subs[country]

        country_pv_optimisations = pv_optimisations[pv_optimisations['location'] == country_clover_sub]
        country_diesel_optimisations = diesel_optimisations[diesel_optimisations['location'] == country_clover_sub]

        # Set empty dfs to append values to as it steps through years
        country_emissions_pv = pd.DataFrame(columns=range(2021, target_year + 1),
                                            index=country_target_cells_number_index.index)
        country_emissions_pv = country_emissions_pv.fillna(0)
        country_investment_pv = pd.DataFrame(columns=range(2021, target_year + 1),
                                             index=country_target_cells_number_index.index)
        country_investment_pv = country_investment_pv.fillna(0)
        country_emissions_diesel = pd.DataFrame(columns=range(2021, target_year + 1),
                                                index=country_target_cells_number_index.index)
        country_emissions_diesel = country_emissions_diesel.fillna(0)
        country_investment_diesel = pd.DataFrame(columns=range(2021, target_year + 1),
                                                 index=country_target_cells_number_index.index)
        country_investment_diesel = country_investment_diesel.fillna(0)

        # Loop through each year
        for year in range(2021, target_year + 1):

            # Closest year needed for pulling out relevant values
            if year in closest_years:
                closest_year = closest_years[year]
            else:
                closest_year = Utils().return_closest_year(target_year, year)
                closest_years[year] = closest_year
            # Extract the year values
            country_year_pv_optimisation = country_pv_optimisations[country_pv_optimisations['year'] == closest_year]
            country_year_diesel_optimisation = country_diesel_optimisations[country_diesel_optimisations['year']
                                                                            == closest_year]

            if system == 'MG':
                division_factor = 100
            elif system == 'SHS':
                division_factor = 1
            else:
                raise Exception("Please set the system to either SHS or MG")

            # Extract the relevant values needed for the calculation of emissions and investment
            years_ahead = 0
            for iteration in range(3):

                appending_to_year = year + years_ahead
                if appending_to_year > target_year:
                    break
                else:
                    iteration_cost_pv = ((country_year_pv_optimisation['iteration_' + str(iteration) +
                                                                       '_total_system_cost'].values) / division_factor)
                    iteration_ghg_pv = (country_year_pv_optimisation['iteration_' + str(iteration) +
                                                                     '_total_system_ghgs'].values) / division_factor
                    iteration_cost_diesel = \
                        (country_year_diesel_optimisation['iteration_' + str(iteration)
                                                          + '_total_system_cost'].values) / division_factor
                    iteration_ghg_diesel = \
                        (country_year_diesel_optimisation['iteration_' + str(iteration)
                                                          + '_total_system_ghgs'].values) / division_factor

                    # Apply additional discounting where needed
                    if year <= LEAF.current_calendar_year:
                        pass
                    else:
                        iteration_cost_pv = Utils().return_discounted_value(iteration_cost_pv, year)
                        iteration_cost_diesel = Utils().return_discounted_value(iteration_cost_diesel, year)

                    # print(country_investment_pv)
                    # print(country_target_cells_number_index[str(year)])
                    # print(iteration_cost_pv)
                    # print(country_info['location'][country], year)
                    # print(system)
                    country_investment_pv[appending_to_year] += country_target_cells_number_index[str(year)] \
                                                                * iteration_cost_pv
                    country_emissions_pv[appending_to_year] += country_target_cells_number_index[str(year)] \
                                                               * iteration_ghg_pv
                    country_investment_diesel[appending_to_year] += country_target_cells_number_index[str(year)] \
                                                                    * iteration_cost_diesel
                    country_emissions_diesel[appending_to_year] += country_target_cells_number_index[str(year)] \
                                                                   * iteration_ghg_diesel
                    years_ahead += 5

        country_investment_pv['id'], country_emissions_pv['id'], country_investment_diesel['id'], \
        country_emissions_diesel['id'] = country_target_cells_number_index['id'], country_target_cells_number_index[
            'id'], \
                                         country_target_cells_number_index['id'], country_target_cells_number_index[
                                             'id']

        pv_investment_out = pd.concat([pv_investment_out, country_investment_pv], axis=0)
        pv_emissions_out = pd.concat([pv_emissions_out, country_emissions_pv], axis=0)
        diesel_investment_out = pd.concat([diesel_investment_out, country_investment_diesel], axis=0)
        diesel_emissions_out = pd.concat([diesel_emissions_out, country_emissions_diesel], axis=0)

    pv_investment_out = pv_investment_out.sort_values(by='id', axis=0, ascending=True)
    pv_emissions_out = pv_emissions_out.sort_values(by='id', axis=0, ascending=True)
    diesel_investment_out = diesel_investment_out.sort_values(by='id', axis=0, ascending=True)
    diesel_emissions_out = diesel_emissions_out.sort_values(by='id', axis=0, ascending=True)

    pv_investment_out = pv_investment_out.reset_index(drop=True)
    pv_emissions_out = pv_emissions_out.reset_index(drop=True)
    diesel_investment_out = diesel_investment_out.reset_index(drop=True)
    diesel_emissions_out = diesel_emissions_out.reset_index(drop=True)

    pv_investment_out = pd.concat([energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
                                   pv_investment_out], axis=1)
    pv_emissions_out = pd.concat([energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
                                  pv_emissions_out], axis=1)

    if LEAF.split_emissions_assets == 'Y':
        print('Splitting emissions by asset lifetime')
        pv_emissions_out.to_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF/Tests/pv_emissions_out_pre.csv')
        pv_emissions_out = Utils().adapt_infrastructure_emissions\
            (pv_emissions_out, LEAF.target_year, LEAF.PV_infrastructure_asset_lifetime)
        pv_emissions_out.to_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF/Tests/pv_emissions_out_post.csv')
    else:
        pass

    diesel_investment_out = pd.concat(
        [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
         diesel_investment_out], axis=1)
    diesel_emissions_out = pd.concat(
        [energy_by_id_year['ADM0_NAME'], energy_by_id_year['ISO3'], energy_by_id_year['id'],
         diesel_emissions_out], axis=1)

    # Add sum column to give the totals
    pv_investment_out['sum'] = pv_investment_out.loc[:, 2021:target_year].sum(axis=1)
    pv_emissions_out['sum'] = pv_emissions_out.loc[:, 2021:target_year].sum(axis=1)
    diesel_investment_out['sum'] = diesel_investment_out.loc[:, 2021:target_year].sum(axis=1)
    diesel_emissions_out['sum'] = diesel_emissions_out.loc[:, 2021:target_year].sum(axis=1)

    # Export files to csv
    pv_investment_out.to_csv(SpatialData.spatial_filepath + 'Off_grid/pv_investment_' + system + '_' + reliability +
                             '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    pv_emissions_out.to_csv(SpatialData.spatial_filepath + 'Off_grid/pv_emissions_' + system + '_' + reliability + '_'
                            + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '.csv')
    diesel_investment_out.to_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_investment_' + system + '_'
                                 + reliability + '_' + scenario + '_' + str(target_year) + '_tier_' + str(
        tier_of_access) + '.csv')
    diesel_emissions_out.to_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_emissions_' + system + '_'
                                + reliability + '_' + scenario + '_' + str(target_year) + '_tier_' + str(
        tier_of_access) + '.csv')

def return_least_cost_option(target_year, scenario, tier_of_access, reliability, reliability_penalty, carbon_prices,
                             reliability_subsidy):
    """
    Function that returns the optimal
    additional features could be added:

        - using the length of grid per grid connected square to add in some additional costs
        - placing a zero on places with no population gaining access
        - placing a unit unmet demand premium on household demand not met

    :param reliability_subsidy: a subsidy for reliability for units of demand supplied above the grid reliability
    :param target_year: the year at which universal electrification is reached
    :param scenario: the scenario being run, 'universal' access
    :param tier_of_access: the demand level being run
    :param reliability: 'grid' level or 'high'
    :return: a dataframe with the least cost option for each cell

    """
    # Import necessary files
    # Existence of grid in cells
    grid_existence = SpatialData.existing_grid_density
    grid_existence_count = SpatialData.existing_grid_density['COUNT']

    # Emissions and investment for national grid per cell
    grid_extension_investment = pd.read_csv(SpatialData.spatial_filepath + 'grid_extension_costs_' +
                                            scenario + '_' + str(target_year) + '_tier_' + str(
        tier_of_access) + '.csv')['sum']

    grid_extension_emissions = pd.read_csv(SpatialData.spatial_filepath + 'total_emissions_grid_' + scenario + '_' +
                                           str(target_year) + '_tier_' + str(tier_of_access) + '.csv')['sum']

    grid_extension_emissions_df = pd.read_csv(SpatialData.spatial_filepath + 'total_emissions_grid_' + scenario + '_' +
                                              str(target_year) + '_tier_' + str(tier_of_access) + '.csv')

    # Emissions and investment for diesel and pv mini-grids
    mini_grid_pv_investment = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_investment_MG_' +
                                          reliability + '_' + scenario + '_' + str(target_year) + '_tier_' +
                                          str(tier_of_access) + '.csv')['sum']

    mini_grid_diesel_investment = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_investment_MG_'
                                              + reliability + '_' + scenario + '_' + str(target_year)
                                              + '_tier_' + str(tier_of_access) + '.csv')['sum']

    mini_grid_pv_emissions = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_emissions_MG_'
                                         + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                         + str(tier_of_access) + '.csv')['sum']

    mini_grid_pv_emissions_df = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_emissions_MG_'
                                            + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                            + str(tier_of_access) + '.csv')

    mini_grid_diesel_emissions = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_emissions_MG_'
                                             + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                             + str(tier_of_access) + '.csv')['sum']

    mini_grid_diesel_emissions_df = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_emissions_MG_'
                                                + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                                + str(tier_of_access) + '.csv')

    # Emissions and investment for diesel and pv stand-alone systems
    stand_alone_pv_investment = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_investment_SHS_' +
                                            reliability + '_' + scenario + '_' + str(target_year) + '_tier_' +
                                            str(tier_of_access) + '.csv')['sum']
    stand_alone_diesel_investment = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_investment_SHS_'
                                                + reliability + '_' + scenario + '_' + str(target_year)
                                                + '_tier_' + str(tier_of_access) + '.csv')['sum']
    stand_alone_pv_emissions = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_emissions_SHS_'
                                           + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                           + str(tier_of_access) + '.csv')['sum']
    stand_alone_pv_emissions_df = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/pv_emissions_SHS_'
                                              + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                              + str(tier_of_access) + '.csv')
    stand_alone_diesel_emissions = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_emissions_SHS_'
                                               + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                               + str(tier_of_access) + '.csv')['sum']
    stand_alone_diesel_emissions_df = pd.read_csv(SpatialData.spatial_filepath + 'Off_grid/diesel_emissions_SHS_'
                                                  + reliability + '_' + scenario + '_' + str(target_year) + '_tier_'
                                                  + str(tier_of_access) + '.csv')

    # Check for inclusion of carbon tax
    if LEAF.global_carbon_tax == 'Y':
        # in sequence pass in each of the cost items along with their respective emissions profiles to give the outputs
        grid_extension_investment += Utils().add_carbon_tax(grid_extension_emissions_df, target_year, carbon_prices)
        mini_grid_pv_investment += Utils().add_carbon_tax(mini_grid_pv_emissions_df, target_year, carbon_prices)
        mini_grid_diesel_investment += Utils().add_carbon_tax(mini_grid_diesel_emissions_df, target_year, carbon_prices)
        stand_alone_pv_investment += Utils().add_carbon_tax(stand_alone_pv_emissions_df, target_year, carbon_prices)
        stand_alone_diesel_investment += \
            Utils().add_carbon_tax(stand_alone_diesel_emissions_df, target_year, carbon_prices)

    if reliability_penalty > 0:

        # Pass the cost items to the lost demand cost function
        if LEAF.reliability == 'grid':
            print('You can only run the unmet demand penalty with high reliability scenario')
            raise ValueError

        else:
            off_grid_penalty, grid_penalty = Utils().add_lost_demand_penalty(tier_of_access, reliability_penalty)
            grid_extension_investment += grid_penalty
            mini_grid_pv_investment += off_grid_penalty
            mini_grid_diesel_investment += off_grid_penalty
            stand_alone_pv_investment += off_grid_penalty
            stand_alone_diesel_investment += off_grid_penalty

    if reliability_subsidy > 0:

        # Raise errors if settings conflicting with reliability subsidy
        if LEAF.reliability == 'grid':
            print('You can only run the unmet demand penalty with high reliability scenario')
            raise ValueError

        if reliability_penalty > 0:
            print('You cannot run both a reliability penalty and subsidy')
            raise ValueError

        # Add the subsidy to the off-grid technologies
        else:
            off_grid_subsidy = Utils().add_met_demand_subsidy(tier_of_access, reliability_subsidy)
            mini_grid_pv_investment += off_grid_subsidy
            mini_grid_diesel_investment += off_grid_subsidy
            stand_alone_pv_investment += off_grid_subsidy
            stand_alone_diesel_investment += off_grid_subsidy

    # Population gaining access
    population_gained_access = pd.read_csv(SpatialData.spatial_filepath + 'No_access_pop/by_year_additions_'
                                           + scenario + '_' + str(target_year) + '.csv')['sum']

    # Set empty variables appended to
    none, grid, mini_grid_pv, mini_grid_diesel, stand_alone_pv, stand_alone_diesel = [], [], [], [], [], []
    mode = []

    # loop through cells and determine 'best' option, appending respective values to lists
    for cell in range(len(grid_existence)):
        """
        Mode Codes:
        0 = no population gaining access
        1 = grid access
        2 = mini-grid pv
        3 = mini-grid diesel
        4 = pv stand alone
        5 = diesel stand alone
        """
        # If there are no people gaining access, append mode as zero.
        if population_gained_access[cell] < 0.5:
            mode.append(0)
        else:
            # If there are grid lines in a given cell, append one, denoting grid access.
            if grid_existence_count[cell] > 0:
                mode.append(1)

            # Otherwise, test based on cost as to what the least-cost option is
            else:
                # Establish if its SHS or mini-grid being compared using the population density
                if population_gained_access[cell] >= LEAF.SHS_density_threshold:
                    pv_off_grid, diesel_off_grid = 2, 3
                    pv_off_grid_comparator = mini_grid_pv_investment[cell]
                    diesel_off_grid_comparator = mini_grid_diesel_investment[cell]
                elif population_gained_access[cell] < LEAF.SHS_density_threshold:
                    pv_off_grid, diesel_off_grid = 4, 5
                    pv_off_grid_comparator = stand_alone_pv_investment[cell]
                    diesel_off_grid_comparator = stand_alone_diesel_investment[cell]
                else:
                    raise Exception("Please enter a value for the SHS density threshold in the model inputs")
                if diesel_off_grid_comparator > grid_extension_investment[cell] < pv_off_grid_comparator:
                    mode.append(1)
                else:
                    if diesel_off_grid_comparator < pv_off_grid_comparator:
                        mode.append(diesel_off_grid)
                    elif pv_off_grid_comparator <= diesel_off_grid_comparator:
                        mode.append(pv_off_grid)


    modes = [none, grid, mini_grid_pv, mini_grid_diesel, stand_alone_pv, stand_alone_diesel]
    for i in range(len(mode)):
        number_code_system = mode[i]
        system_choice = modes[number_code_system]
        system_choice.append(1)
        for j in range(len(modes)):
            if modes[j] != system_choice:
                modes[j].append(0)

    # Make dataframe of outputs
    output_df = pd.DataFrame([])
    output_df['id'], output_df['ISO3'], output_df['ADM0_NAME'] = \
        grid_existence['id'], grid_existence['ISO3'], grid_existence['ADM0_NAME']

    # Set Columns with modes of each system
    output_df['grid'], output_df['mini_grid_pv'], output_df['mini_grid_diesel'] = grid, mini_grid_pv, mini_grid_diesel
    output_df['SA_pv'], output_df['SA_diesel'] = stand_alone_pv, stand_alone_diesel

    # Calculate national grid investment and emissions
    output_df['grid_investment'], output_df['grid_emissions'] = \
        grid_extension_investment * output_df['grid'], grid_extension_emissions * output_df['grid']

    # Calculate pv mini-grid investment and emissions
    output_df['pv_mini_grid_investment'], output_df['pv_mini_grid_emissions'] = \
        mini_grid_pv_investment * output_df['mini_grid_pv'], mini_grid_pv_emissions * output_df['mini_grid_pv']

    # Calculate diesel mini-grid investment and emissions
    output_df['diesel_mini_grid_investment'], output_df['diesel_mini_grid_emissions'] = \
        mini_grid_diesel_investment * output_df['mini_grid_diesel'], mini_grid_diesel_emissions * \
        output_df['mini_grid_diesel']

    # Calculate pv stand-alone system investment and emissions
    output_df['pv_sa_investment'], output_df['pv_sa_emissions'] = \
        stand_alone_pv_investment * output_df['SA_pv'], stand_alone_pv_emissions * output_df['SA_pv']

    # Calculate diesel stand-alone system investment and emissions
    output_df['diesel_sa_investment'], output_df['diesel_sa_emissions'] = \
        stand_alone_diesel_investment * output_df['SA_diesel'], stand_alone_diesel_emissions * output_df['SA_diesel']

    # Set mode column for use in plotting
    output_df['mode'] = mode

    # Calculate total emissions and investment
    output_df['total_investment'] = \
        output_df['diesel_mini_grid_investment'] + output_df['pv_mini_grid_investment'] + output_df['grid_investment'] \
        + output_df['diesel_sa_investment'] + output_df['pv_sa_investment']

    output_df['total_emissions'] = \
        output_df['pv_mini_grid_emissions'] + output_df['diesel_mini_grid_emissions'] + output_df['grid_emissions'] \
        + output_df['diesel_sa_emissions'] + output_df['pv_sa_emissions']

    # Calculate total population met by each means
    output_df['pv_mini_grid_population'] = population_gained_access * output_df['mini_grid_pv']
    output_df['diesel_mini_grid_population'] = population_gained_access * output_df['mini_grid_diesel']
    output_df['diesel_SA_population'] = population_gained_access * output_df['SA_diesel']
    output_df['pv_SA_population'] = population_gained_access * output_df['SA_pv']
    output_df['grid_population'] = population_gained_access * output_df['grid']

    summary_sheet = Utils().get_summary_statistics(output_df)
    summary_sheet = pd.DataFrame.from_dict(summary_sheet, orient='columns')
    country_summary_sheet = Utils().country_summary_stats(output_df)

    if LEAF.sensitivity_analysis == 'Y':
        return summary_sheet, output_df
    elif LEAF.sensitivity_analysis == 'N':

        if reliability_subsidy > 0:
            # Export summary sheets to csv file
            summary_sheet.to_csv(SpatialData.spatial_filepath + 'cost_optimal_output_summary_' + reliability +
                                 '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                                 + LEAF.global_carbon_tax + '_RS' + str(LEAF.met_demand_subsidy) + '.csv')
            country_summary_sheet.to_csv(
                SpatialData.spatial_filepath + 'cost_optimal_output_countries_summary_' + reliability +
                '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                + LEAF.global_carbon_tax + '_RS' + str(LEAF.met_demand_subsidy) + '.csv')

            # Export by cell outputs to file
            output_df.to_csv(SpatialData.spatial_filepath + 'cost_optimal_output_by_cell_' + reliability +
                             '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                             + LEAF.global_carbon_tax + '_RS' + str(LEAF.met_demand_subsidy) + '.csv')

        else:

            # Export summary sheets to csv file
            summary_sheet.to_csv(SpatialData.spatial_filepath + 'cost_optimal_output_summary_' + reliability +
                                 '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                                 + LEAF.global_carbon_tax + '_RP' + str(LEAF.unmet_demand_penalty) + '.csv')
            country_summary_sheet.to_csv(SpatialData.spatial_filepath + 'cost_optimal_output_countries_summary_' + reliability +
                                 '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                                 + LEAF.global_carbon_tax + '_RP' + str(LEAF.unmet_demand_penalty) + '.csv')

            # Export by cell outputs to file
            output_df.to_csv(SpatialData.spatial_filepath + 'cost_optimal_output_by_cell_' + reliability +
                             '_' + scenario + '_' + str(target_year) + '_tier_' + str(tier_of_access) + '_CT_'
                             + LEAF.global_carbon_tax + '_RP' + str(LEAF.unmet_demand_penalty) + '.csv')


def population_summaries():
    population_summary = pd.DataFrame([])
    countries = LEAF.country_data_sheet['ISO3']
    population = SpatialData.population_by_year
    population_indexed = population.set_index('ISO3')
    population_summary['ISO3'] = LEAF.country_data_sheet['ISO3']
    population_summary['Country'] = LEAF.country_data_sheet['location']

    # Loop through each year
    for year in range(2020, 2036):

        year_totals = []

        # Loop through countries and add up
        for country in countries:
            country_cells = population_indexed.loc[country]
            year_totals.append(np.sum(country_cells[str(year)]))

        population_summary[str(year)] = year_totals

    population_summary.to_csv(LEAF.model_outputs + 'population_totals_by_year.csv')


if __name__ == "__main__":
    main()
