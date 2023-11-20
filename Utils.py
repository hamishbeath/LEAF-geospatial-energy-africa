import pandas as pd
import numpy as np


# from main import *
# from main import SEAR, SpatialData


class Utils:
    clover_results_filepath = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA/Results/'
    external_load_filepath = '/Volumes/Hamish_ext/Mitigation_project/CLOVER_inputs/Load/'


    def return_relevant_clover_results_summary(self, ssp, tier, system):

        # Import relevant file of results
        raw_results = pd.read_csv(Utils.clover_results_filepath + 'SSP' + str(ssp) + '_' + system + '.csv',
                                  index_col=None)

        # Filter by selected tier
        tier_results = raw_results[raw_results['Tier'] == tier]

        # Create filter to relevant scenarios
        pv_full_reliability = tier_results[tier_results['index'] == 'optimisation_output_1.json']
        diesel_full_reliability = tier_results[tier_results['index'] == 'optimisation_output_2.json']
        pv_grid_reliability = tier_results[tier_results['index'] == 'optimisation_output_3.json']
        diesel_grid_reliability = tier_results[tier_results['index'] == 'optimisation_output_4.json']

        # Output to respective .csv files
        pv_full_reliability.to_csv(Utils.clover_results_filepath + 'pv_full_reliability_SSP_' + str(ssp) +
                                   '_tier_' + str(tier) + '_' + system + '.csv')
        diesel_full_reliability.to_csv(Utils.clover_results_filepath + 'diesel_full_reliability_SSP_' +
                                       str(ssp) + '_tier_' + str(tier) + '_' + system + '.csv')
        pv_grid_reliability.to_csv(Utils.clover_results_filepath + 'pv_grid_reliability_SSP_' + str(ssp)
                                   + '_tier_' + str(tier) + '_' + system + '.csv')
        diesel_grid_reliability.to_csv(Utils.clover_results_filepath + 'diesel_grid_reliability_SSP_'
                                       + str(ssp) + '_tier_' + str(tier) + '_' + system + '.csv')

        return pv_full_reliability, diesel_full_reliability, pv_grid_reliability, diesel_grid_reliability


    def return_closest_year(self, target_year,year):
        """
           Util Function that returns the closest year the five-year arrays to be appended to dict
           Inputs: year, end year of simulation
           Outputs: The closest year to the input year
        """
        years = [*range(2020, target_year + 1, 5)]
        years = np.asarray(years)
        idx = (np.abs(years - year)).argmin()
        return years[idx]


    def return_summary_sheet_years_countries(self, target_year, input_df):
        """
        Util function that returns a summary dataframe that provides by-country summaries of input df
        Inputs: target year, input dataframe
        Outputs: summary dataframe with years, countries and sum total
        """
        # Set output dataframe
        output_df = pd.DataFrame([])
        countries = SEAR.country_data_sheet['ISO3']

        output_df['IS03'] = countries

        # Make sum empty column
        output_df['sum'] = np.zeros(len(countries))

        # Work through each year in the simulation
        for year in range(2021, target_year + 1):

            # Create empty list to append to
            year_totals = []

            # Loop through countries and add up
            for country in countries:
                country_cells = input_df[input_df['ISO3'] == country]
                year_totals.append(np.sum(country_cells[str(year)]))

            # Append to necessary columns in dataframe
            output_df[str(year)] = year_totals
            output_df['sum'] += year_totals

        return output_df

    def return_discounted_value(self, input_value, year):

        """
        Util function that returns a discounted value of a value fed into it from mini-grid costings. These values have
        already been discounted for non-year zero costs so are discounted to an additional amount guided by the year
        and present year.
        Inputs: Input_value, year in cost calculation
        Outputs: Discounted value
        """
        reduction_factor = year - SEAR.current_calendar_year
        output_value = input_value * (1.0 - SEAR.discount_rate) ** reduction_factor
        return output_value

    # Util function that adds carbon tax
    def add_carbon_tax(self, emissions, target_year, carbon_prices):
        """
        Util function that calculates the carbon tax cost from the input emissions profile
        :param emissions: a dataframe that has the emissions by cell and by year of simulation
        :param target_year: the target year for universal access
        :return: a df column by cell of th discounted and summed carbon tax costs
        """
        emissions['tax_sum'] = np.zeros(len(emissions))
        for year in range(2021, target_year + 1):
            try:
                year_emissions = emissions[str(year)]
            except KeyError:
                year_emissions = emissions[year]
            carbon_tax_year = Utils.return_closest_year(self, target_year, year)
            carbon_tax = carbon_prices[carbon_tax_year]
            year_tax = (year_emissions / 1000) * carbon_tax
            year_tax = Utils.return_discounted_value(self, year_tax, year)
            emissions['tax_sum'] += year_tax
        return emissions['tax_sum']


    # Util function that adapts emissions profiles for solar infrastructure and distribution network
    def adapt_infrastructure_emissions(self, emissions, target_year, asset_lifetime):
        """
        Util function designed to spread the emissions occurring from certain types of infrastructure over the
        asset lifetime of the infrastructure. This is done by dividing the emissions by the asset lifetime and
        applying it for each year remaining of the simulation.
        """
        # Create dataframe of zeros with length of emissions with columns for each year 2021 to target year
        empty_df = pd.DataFrame(np.zeros((len(emissions), target_year - 2020)))
        empty_df.columns = [*range(2021, target_year + 1)]
        year = 2021
        for column in range(2021, target_year + 1):
            try:
                emissions_year = emissions[str(column)]
            except KeyError:
                emissions_year = emissions[column]
            emissions_year = emissions_year / asset_lifetime
            for year_remaining in range(year, target_year + 1):
                empty_df[year_remaining] += emissions_year
            year += 1
        # Concat several columns from original emission df to new one
        output_df = pd.concat([emissions['id'],emissions['ADM0_NAME'],emissions['ISO3'], empty_df], axis=1)

        # Return output df to main function
        return output_df

    # Util function that returns summary stats at the country level
    def country_summary_stats(self, input_df):

        """
        Function that returns summary stats for a given scenario (input) by country for use in
        country level analysis and plotting.

        :param input_df: input dataframe with information by cell
        :return: by-country insights
        """
        # Set output dataframe for summaries by country
        countries_summaries_df = pd.DataFrame([])

        # Loop through countries
        for country in range(0, len(SEAR.country_data_sheet)):

            country_select = SEAR.country_data_sheet['ISO3'][country]
            cells_select = input_df.loc[input_df['ISO3'] == country_select]
            country_summary = Utils().get_summary_statistics(cells_select)
            country_summary_df = pd.DataFrame.from_dict(country_summary, orient='columns')
            countries_summaries_df = pd.concat([countries_summaries_df, country_summary_df], axis=0)

        countries_summaries_df = countries_summaries_df.reset_index(drop=True)
        countries_summaries_df['ISO3'] = SEAR.country_data_sheet['ISO3']
        countries_summaries_df['Country'] = SEAR.country_data_sheet['location']

        return countries_summaries_df

    # Util function that returns the summary statistics for a given dataframe
    def get_summary_statistics(self, input_df):
        """
        Function that returns summary statistics of the output df
        :param input_df: the dataframe of per cell outputs fed to the function
        :return: a summary csv of the by-cell outputs
        """
        modes = ['pv_mini_grid', 'diesel_mini_grid', 'diesel_SA', 'pv_SA', 'grid']
        output_df = {}

        # Calculate values needed to outputs
        total_pv_mini_grid_population = np.sum(input_df['pv_mini_grid_population'].values)
        total_diesel_mini_grid_population = np.sum(input_df['diesel_mini_grid_population'].values)
        total_diesel_sa_population = np.sum(input_df['diesel_SA_population'].values)
        total_pv_sa_population = np.sum(input_df['pv_SA_population'].values)
        total_grid_population = np.sum(input_df['grid_population'].values)
        total_population = \
            total_pv_mini_grid_population + total_diesel_mini_grid_population + \
            total_diesel_sa_population + total_pv_sa_population + total_grid_population

        # Set columns as values in output frame
        output_df['total_population'] = [total_population]
        output_df['total_pv_mini_grid_population'] = [total_pv_mini_grid_population]
        output_df['total_diesel_mini_grid_population'] = [total_diesel_mini_grid_population]
        output_df['total_diesel_SA_population'] = [total_diesel_sa_population]
        output_df['total_pv_SA_population'] = [total_pv_sa_population]
        output_df['total_grid_population'] = [total_grid_population]

        # Loop through modes to produce further output columns
        for mode in modes:
            mode_pop_percentage = output_df['total_' + mode + '_population'] / total_population
            output_df[mode + '_percentage_pop'] = mode_pop_percentage[0]

        return output_df

    # Util function that adds the lost demand penalty to the costs
    def add_lost_demand_penalty(self, tier, reliability_penalty):

        # Import electricity demand
        electricity_demand_by_cell = \
            pd.read_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_' + SEAR.scenario + '_' +
                        str(SEAR.target_year) + '_tier_' + str(tier) + '.csv')

        # Create high-reliability cost addition
        high_rel = pd.DataFrame(electricity_demand_by_cell['id'])
        high_rel['sum'] = np.zeros(len(high_rel))
        for year in range(2021, SEAR.target_year + 1):
            year_column = pd.DataFrame(electricity_demand_by_cell[str(year)])
            year_column = year_column * 0.1 * reliability_penalty
            year_column_discounted = Utils.return_discounted_value(self, year_column, year)
            high_rel['sum'] += year_column_discounted[str(year)]

        # Calculate grid-reliability cost additions
        grid_rel = pd.DataFrame([])
        for i in range(0, len(SEAR.country_data_sheet)):
            country = SEAR.country_data_sheet['ISO3'][i]
            penalty_factor = SEAR.country_data_sheet['Blackout Threshold'][i]
            country_cells = pd.DataFrame(electricity_demand_by_cell.loc[electricity_demand_by_cell['ISO3'] == country])
            output_df = pd.DataFrame(country_cells['id'])
            output_df['sum'] = np.zeros(len(country_cells))
            for year in range(2021, SEAR.target_year + 1):
                year_country_penalty = country_cells[str(year)] * penalty_factor * reliability_penalty
                year_country_penalty = Utils.return_discounted_value(self, year_country_penalty, year)
                output_df['sum'] += year_country_penalty
            grid_rel = pd.concat([grid_rel, output_df], axis=0)

        grid_rel = grid_rel.sort_values(by='id', axis=0, ascending=True)
        return high_rel['sum'], grid_rel['sum']

    # Util function that adds 'met demand' subsidy above grid reliability
    def add_met_demand_subsidy(self, tier, reliability_subsidy):

        # Import electricity demand
        electricity_demand_by_cell = \
            pd.read_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_' + SEAR.scenario + '_' +
                        str(SEAR.target_year) + '_tier_' + str(tier) + '.csv')

        # Calculate off-grid subsidy based on additional demand met above grid
        grid_rel = pd.DataFrame([])
        for i in range(0, len(SEAR.country_data_sheet)):
            country = SEAR.country_data_sheet['ISO3'][i]
            subsidy_factor = (SEAR.country_data_sheet['Blackout Threshold'][i]) - 0.1
            print(country, subsidy_factor)
            country_cells = pd.DataFrame(electricity_demand_by_cell.loc[electricity_demand_by_cell['ISO3'] == country])
            output_df = pd.DataFrame(country_cells['id'])
            output_df['sum'] = np.zeros(len(country_cells))
            for year in range(2021, SEAR.target_year + 1):
                year_country_subsidy = country_cells[str(year)] * subsidy_factor * (- reliability_subsidy)
                year_country_subsidy = Utils.return_discounted_value(self, year_country_subsidy, year)
                output_df['sum'] += year_country_subsidy
            grid_rel = pd.concat([grid_rel, output_df], axis=0)


        subsidy = grid_rel.sort_values(by='id', axis=0, ascending=True)
        return subsidy['sum']



    # Util function that categorises countries by demand growth and calculates the average to peak load factor
    def categorise_countries_by_demand_growth(self, end_year):

        # Set empty df to append average / peak
        peak_factors = pd.DataFrame()
        peak_factors['Countries'], peak_factors['ISO3'] = \
            SEAR.country_data_sheet['location'], SEAR.country_data_sheet['ISO3']
        # Set empty list to give the calculated tiers
        totals_tiers = np.zeros(len(SEAR.country_data_sheet))
        for tier in range(1, 5):

            # loop through countries and calculate demand difference in start and end year,
            # and mean vs peak demand factor
            country_growth_factors = []
            peak_load_factors = []
            for i in range(0, len(SEAR.country_energy_subs)):
                country = SEAR.country_energy_subs['location'][i]
                energy_sub = SEAR.country_energy_subs['energy_file'][i]
                # Import electricity demand for country
                start_year_file = \
                    pd.read_csv(self.external_load_filepath + 'Load_by_year/SSP' + str(SEAR.ssp) + '/Tier ' +
                    str(tier) + '/' + energy_sub + '_community_load_2020.csv', index_col=0)['2020']
                end_year_file = \
                    pd.read_csv(self.external_load_filepath + 'Load_by_year/SSP' + str(SEAR.ssp) + '/Tier ' +
                    str(tier) + '/' + energy_sub + '_community_load_' + str(end_year) + '.csv',
                                       index_col=0)[str(end_year)]
                start_year_sum = np.sum(start_year_file.values)
                end_year_sum = np.sum(end_year_file.values)
                demand_growth_factor = end_year_sum / start_year_sum
                country_growth_factors.append(demand_growth_factor)
                peak_load_factor = np.max(end_year_file) / np.mean(end_year_file)
                peak_load_factors.append(peak_load_factor)

            totals_tiers += country_growth_factors
            peak_factors['Tier_' + str(tier)] = peak_load_factors

        totals_tiers = totals_tiers / 4
        # Categorise countries by demand growth
        categories = pd.DataFrame(SEAR.country_data_sheet['ISO3'])
        categories['demand_growth_factor'] = totals_tiers
        peak_factors.to_csv(SEAR.model_inputs + 'grid/grid_peak_factors.csv')
        categories.to_csv(SEAR.model_inputs + 'demand_growth_factor.csv')

    # Util function that calculates grid coverage by country
    def calculate_grid_coverage_by_country(self):

        existing_grid_distance = pd.read_csv(SpatialData.spatial_filepath + 'existing_grid_length.csv')
        existing_grid_density = pd.read_csv(SpatialData.spatial_filepath + 'existing_grid_density.csv')
        households_by_fid = pd.read_csv(SpatialData.spatial_filepath + 'households_added_by_year_universal_2030.csv')
        countries_pop_weighted_average_dist = []
        countries_grid_density = []
        countries_list = []
        # Loop through countries to calculate average distance to grid and total length of grid lines
        for i in range(0, len(SEAR.country_data_sheet)):

            country = SEAR.country_data_sheet['ISO3'][i]
            area = SEAR.country_data_sheet['area'][i]

            # Calculate household-weighted average distance to grid
            country_cells_distance = pd.DataFrame(existing_grid_distance.loc[existing_grid_distance['ISO3'] == country])
            distance_to_grid = country_cells_distance['NEAR_DIST']
            country_households = pd.DataFrame(households_by_fid.loc[households_by_fid['ISO3'] == country])
            country_households['distance_to_grid'] = distance_to_grid
            country_households = country_households.loc[country_households['sum'] > 0]
            country_households_sum = country_households['sum']
            summed_country_households = np.sum(households_by_fid['sum'])
            countries_list.append(country)
            country_households['sum'] = country_households['sum'] / summed_country_households
            country_households['proportion_of_total'] = country_households_sum / summed_country_households
            country_households['weighted_distance'] = country_households['proportion_of_total'] * distance_to_grid
            mean_pop_weighted_distance = np.mean(country_households['weighted_distance'])
            countries_pop_weighted_average_dist.append(mean_pop_weighted_distance)

            # Calculate grid density (km of grid per km2 of country)
            country_cells_density = pd.DataFrame(existing_grid_density.loc[existing_grid_density['ISO3'] == country])
            total_grid_length = np.sum(country_cells_density['LENGTH'])
            countries_grid_density.append(total_grid_length / area)

        # Export results to csv
        output_df = pd.DataFrame()
        output_df['ISO3'] = countries_list
        output_df['average_pop_weighted_dist'] = countries_pop_weighted_average_dist
        output_df['grid_density'] = countries_grid_density
        output_df.to_csv(SEAR.model_inputs + 'grid_coverage_by_country.csv')

    # Util function that provides mean reliability of a scenario, and emissions in 2030 year
    def scenario_analysis(self, tier, scenario, mode):
        rel = 'full'
        # Import relevant scenario
        scenario_df = pd.read_csv(SpatialData.spatial_filepath + '/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_'
                               + str(tier) + '_' + str(scenario) + '.csv')
        energy_df = \
            pd.read_csv(SpatialData.spatial_filepath + 'energy_demand_by_year_universal_2030_tier_' + str(tier) + '.csv')

        # calculate total energy demand across all years 2021-2030
        total_energy_demand = np.sum(energy_df.loc[:,'2021':'2030'].values)
        print(total_energy_demand)
        # Set empty dictionary to store country reliability and energy shares
        country_reliability_shares = {}
        total_output = []

        if mode == 'all':

            country_data = pd.read_csv(SEAR.model_inputs + 'country_inputs.csv')
            for i in range(0, len(SEAR.country_data_sheet)):
                country = SEAR.country_data_sheet['ISO3'][i]
                reliability = SEAR.country_data_sheet['Rural_energy_met_dec'][i]
                country_cells = scenario_df.loc[scenario_df['ISO3'] == country]
                energy_cells = energy_df.loc[energy_df['ISO3'] == country]
                summed = energy_cells.loc[:,'2021':'2030'].sum(axis=1)

                # Calculate the total energy demand for the country
                country_energy_demand = np.sum(energy_cells.loc[:,'2021':'2030'].values)
                demand_share = country_energy_demand / total_energy_demand

                # Calculate the grid share and non grid share for the country
                grid_energy = country_cells['grid'] * summed
                grid_share = np.sum(grid_energy.values) / country_energy_demand
                country_factor = (grid_share * reliability) + ((1 - grid_share) * 0.9)
                # country_reliability_shares[country] = country_factor
                # Append country reliability and energy share to dictionary
                country_weighted_amount = country_factor * demand_share
                # Filter out countries with no energy demand
                if country_weighted_amount > 0:
                    total_output.append(country_weighted_amount)
                # country_reliability_shares[country_factor] = demand_share

        elif mode == 'grid':

            country_data = pd.read_csv(SEAR.model_inputs + 'country_inputs.csv')
            for i in range(0, len(SEAR.country_data_sheet)):
                country = SEAR.country_data_sheet['ISO3'][i]
                reliability = SEAR.country_data_sheet['Rural_energy_met_dec'][i]
                # country_cells = scenario_df.loc[scenario_df['ISO3'] == country]
                energy_cells = energy_df.loc[energy_df['ISO3'] == country]
                country_energy_demand = np.sum(energy_cells.loc[:, '2021':'2030'].values)
                demand_share = country_energy_demand / total_energy_demand
                country_weighted_amount = reliability * demand_share
                if country_weighted_amount > 0:
                    total_output.append(country_weighted_amount)

        print(np.sum(total_output))

    # Util function that calculates the cost and capacity factor of additional grid capacity for each country
    def grid_cost_and_cf(self):
        # Import required files
        cost_cf_main = pd.read_csv(SEAR.model_inputs + 'grid/grid_cf_costs.csv', index_col=['Mode'])
        grid_mode_shares = pd.read_csv(SEAR.model_inputs + 'grid/grid_electricity_shares.csv', index_col=['Mode'])
        renewables = ['PV', 'Hydro', 'Wind']
        country_CFs = {}
        country_costs = {}

        # loop through each country
        for i in range(0, len(SEAR.country_data_sheet)):

            country_name = SEAR.country_data_sheet['location'][i]
            country_iso = SEAR.country_data_sheet['ISO3'][i]
            country_shares = grid_mode_shares[country_name]

            country_cost = 0
            country_cf = 0

            # loop through each row to establish mode
            for row in range(country_shares.shape[0]):
                mode = country_shares.index[row]
                share = country_shares[row]
                if float(share) > 0:
                    # if the mode is renewables, then find the relevant country capacity factor
                    if mode in renewables:
                        current_mode_cf = Utils().look_up_renewables_cf(mode, country_iso)
                        current_mode_cost = int(cost_cf_main.loc[mode, 'Cost'])
                        country_cost += current_mode_cost * share
                        country_cf += current_mode_cf * share

                    # Otherwise look-up the capacity factor in the table
                    else:
                        current_mode_cf = float(cost_cf_main.loc[mode, 'CF'])
                        current_mode_cost = int(cost_cf_main.loc[mode, 'Cost'])
                        country_cost += current_mode_cost * share
                        country_cf += current_mode_cf * share
                    # Calculate the weighted average capacity factor and cost
                else:
                    pass

            country_CFs[country_name] = country_cf
            country_costs[country_name] = country_cost

        # Export to csv
        output_df = pd.DataFrame.from_dict(country_CFs, orient='index')
        output_df.columns = ['CF']
        output_df['Cost'] = pd.DataFrame.from_dict(country_costs, orient='index')
        output_df['ISO3'] = SEAR.country_data_sheet['ISO3']
        output_df.to_csv(SEAR.model_inputs + 'grid/grid_cf_costs_by_country.csv')

    # Sub-function that returns the relevant renewables mode capacity factor for a country
    def look_up_renewables_cf(self, mode, country):

        # Import required files
        cf_renewables = pd.read_csv(SEAR.model_inputs + 'grid/grid_rn_cf.csv')
        country_cf = cf_renewables.loc[cf_renewables['ISO3'] == country][mode].values[0]
        return country_cf

    # Function that returns the cost of the additional generating capacity required by cell
    def grid_capacity_cost(self, input_demand_df, tier_of_access):

        # Import required files
        grid_costs = SEAR.country_data_sheet['Grid_cost']
        grid_cf = SEAR.country_data_sheet['Grid_CF']
        peak_factors = pd.read_csv(SEAR.model_inputs + 'grid/grid_peak_factors.csv')['Tier_' + str(tier_of_access)]
        rural_energy_met = SEAR.country_data_sheet['Rural_energy_met_dec']
        # Output df
        output_df = pd.DataFrame()

        for country in range(0, len(SEAR.country_data_sheet)):

            country_iso = SEAR.country_data_sheet['ISO3'][country]
            country_grid_cost = grid_costs[country]
            country_grid_cf = grid_cf[country]
            country_peak_factor = peak_factors[country]
            capacity_required = 1 / country_grid_cf
            country_demand = input_demand_df.loc[input_demand_df['ISO3'] == country_iso]
            # Factor in reliability
            total_country_demand_reduced = country_demand[str(SEAR.target_year)] * rural_energy_met[country]

            # Divide by 8760 to get hourly demand - but add peak load factor!
            adjusted_hourly_demand_by_cell = total_country_demand_reduced / 8760

            # Calculate the additional capacity required, by cell
            additional_capacity_by_cell = adjusted_hourly_demand_by_cell * capacity_required * country_peak_factor
            additional_capacity_cost_by_cell = additional_capacity_by_cell * country_grid_cost

            # Add to output df
            country_df = pd.DataFrame(additional_capacity_cost_by_cell)
            country_df.columns = ['Grid_cost']
            country_df['ISO3'] = country_demand['ISO3']
            country_df['id'] = country_demand['id']

            output_df = pd.concat([output_df, country_df], axis=0)

        output_df = output_df.sort_values(by='id', axis=0, ascending=True)
        return output_df['Grid_cost']

    # Function that calculates the NPC of the household installation costs
    def household_installation_cost(self, target_year):

        # Import households added per year and installation costs
        households_added \
            = pd.read_csv(SpatialData.spatial_filepath + 'households_added_by_year_' + SEAR.scenario + '_' +
                     str(target_year) + '.csv')
        installation_cost = SEAR.install_cost
        # Set output df
        output_df = pd.DataFrame()
        output_df['ISO3'] = households_added['ISO3']
        output_df['id'] = households_added['id']
        # make column of zeros
        output_df['total'] = np.zeros(len(households_added))
        print(output_df['total'])
        for i in range(2021, target_year + 1):
            year_install_costs_by_cell = households_added[str(i)] * installation_cost
            if i > 2022:
                year_install_costs_by_cell = Utils().return_discounted_value(year_install_costs_by_cell, i)
            output_df[str(i)] = year_install_costs_by_cell
            output_df['total'] += year_install_costs_by_cell
        output_df.to_csv(SpatialData.spatial_filepath + 'household_installation_costs_' + SEAR.scenario + '_' + str(target_year) + '.csv')

    # Util Function that simplifies mode column to only three categories
    def simplify_mode(self, mode_df):
        """
        :param mode_df: dataframe with mode column to be simplified
        :return: simplified mode column with only four categories as opposed to six
        """
        # 1 Grid, 2 PV, 3 Diesel
        mode_column = mode_df['mode']
        array = []
        for cell in range(0, len(mode_column)):
            if mode_column[cell] == 4:
                array.append(2)
            elif mode_column[cell] == 5:
                array.append(3)
            else:
                array.append(mode_column[cell])
        mode_df['mode_simplified'] = array
        return mode_df['mode_simplified']

    # Util function that isolates layer based on required values and two input layers
    def isolate_layer(self, reference_layer, input_layer, reference_value, scenario_value):

        mode = reference_layer['mode_simplified']
        mode_diff = input_layer['mode_diff']
        array = []
        for cell in range(0, len(mode)):
            if mode[cell] == reference_value and mode_diff[cell] == scenario_value:
                array.append(1)
            else:
                array.append(np.nan)
        input_layer['isolate'] = array
        return input_layer['isolate']
