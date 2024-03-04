# from main import *
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import geopandas as gp
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from main import LEAF

from Utils import *
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

class FishNet:
    geo_file_path = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA/SPATIAL/fishnet_planar/Geo/'
    plotting_output_filepath = 'Outputs/plotting_materials/'

class Plotting:

    home_directory = home_directory = 'hrb16' # 'hamishbeath' # 'hrb16' 
    plotting_output = 'Outputs/plotting_materials/'

# class SpatialPlotting:

    # energy_by_fid = pd.read_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
    #                         '/SEAR_A/Spatial/energy_by_fid_SSP2_min_tier3.csv')
    # grid_cost_emissions = pd.read_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
    #                         '/SEAR_A/Spatial/grid_cost_emissions_tier2_ssp2.csv')
    #mg_cost_emissions = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial/MG_Emissions_Costs.csv')
    # #no_access_growth = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial/no_access_pop_growth_SSP3.csv')
    # #processed_pop_growth_access = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
    #                                           '/Spatial/No_access_pop/no_access_processed_snapshot_2030_SSP2.csv')

def main() -> None:

    # add_extra_fid_cols = pd.read_csv('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
    #                                  '/DATA/SPATIAL/fishnet/Grid/centroids_fuller_join_fids.csv')
    # filename = 'OUTPUT_centroids_joined_fids_fuller'
    # join_attribute_centroids(add_extra_fid_cols, filename)

    # Join attributes for plotting
    # input_csv = pd.read_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
    #                         '/SEAR_A/Spatial/population_by_year.csv')
    # input_filename = 'population_by_year_test'
    # join_attribute(input_csv, input_filename)
    # tree_maps()
    # plot_subplot_maps()
    # plot_subplot_maps_three_cat()
    # plot_map_categories()
    # plot_map_divergant()
    mapped_sensitivity_rel(2)
    # for i in range(2, 5):
    #     plot_scenario_shift_map('CT_Y_RP0.0', i, 'full')
    # simple_population_density_plot()
    # emissions_change_map(4)
    # plot_heatmaps_with_icons()
    # plot_sensitivity()
    # plot_africa_choropleth()
    # mean_reliability_mapped(2, 'CT_N_RP0.5')
    # mean_reliability_diff_mapped(2, 'CT_N_RP0.5', 'full')
    # for i in range(4, 5):
    #     mapped_sensitivity_carbon(i)
    # investment_requirement_mapped(3, 'CT_N_RP0.0')
    # mapped_sensitivity_carbon(4)
    

# Function for plotting country level data on a map
def plot_africa_choropleth():

    #Import Africa shapefile:
    africa_shp = gp.read_file('Geo/PLOTTING_Export_Output_africa_bound.shp')
    africa_stats = LEAF.country_data_sheet
    map_and_stats = africa_shp.merge(africa_stats, on='ISO3')
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    # ax1 = map_and_stats.plot(column="diesel_sub", cmap='Reds', linewidth=0.5, ax=ax1, edgecolor=".4", legend=True)
    ax1 = map_and_stats.plot(column="Rural_energy_met_dec", cmap='hot', linewidth=0.5, ax=ax1, edgecolor=".4", legend=True, vmin=0.3)
    # ax1 = map_and_stats.plot(column="Grid Emissions Intensity (kgCO2/kWh)", cmap='Reds', linewidth=0.5, ax=ax1, edgecolor=".4",legend=True)
    # ax4 = map_and_stats.plot(column="demand_growth_factor", cmap='hot_r', linewidth=0.5, ax=ax4, edgecolor=".4", legend=True) #legend=True, #legend_loc='lower center')
    # ax1 = map_and_stats.plot(column="diesel_sub", cmap='hot_r', linewidth=0.5, ax=ax1, edgecolor=".4", legend=True)
    # ax2 = map_and_stats.plot(column="Rural_energy_met_dec", cmap='hot', linewidth=0.5, ax=ax2, edgecolor=".4", legend=True, vmin=0.3)
    # ax3 = map_and_stats.plot(column="Grid Emissions Intensity (kgCO2/kWh)", cmap='hot_r', linewidth=0.5, ax=ax3, edgecolor=".4",legend=True)
    # ax4 = map_and_stats.plot(column="demand_growth_factor", cmap='hot_r', linewidth=0.5, ax=ax4, edgecolor=".4", legend=True) #legend=True, #legend_loc='lower center')

    # ax5 = map_and_stats.plot(column="grid_density", cmap='hot', linewidth=0.5, ax=ax5, edgecolor=".4", legend=True)
    # ax1.set_title('Diesel Prices (2022 USD/litre)')
    ax1.set_title('National Grid Est. % Rural Energy Met')
    # ax3.set_title('Grid Emissions Intensity (kgCO2/kWh')
    # ax4.set_title('Grid Demand Growth Factor')
    # ax5.set_title('Population-weighted mean distance from grid (km)')
    ax1.axis('off')
    
    # ax2.axis('off')
    # ax3.axis('off')
    # ax4.axis('off')
    # # ax5.axis('off')
    # place all legends from each axis at bottom of subplots
    plt.savefig('Outputs/Plotting_outputs/' + 'heatmap_country_grid_energy_met.pdf')
    plt.tight_layout()
    plt.show()

# Function for plotting the sensitivity analysis of the carbon price and the reliability penalty/ subsidy
def plot_sensitivity():

    plt.rcParams['font.size'] = 12
    # pl.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    #pl.rcParams['ytick.labelright'] = True
    #pl.rcParams['ytick.major.size'] = 0
    #pl.rcParams['ytick.major.pad'] = -56
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
    
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    # set seaborn theme
    sns.set_theme(style="whitegrid")
    line_width = 1.5
    for i in range(2, 5):
        tier_reliability = \
            pd.read_csv('Outputs/Sensitivity/Reliability_penalty_sensitivity_analysis_tier' + str(i) + '.csv', index_col=0)
        # tier_reliability = \
        #          pd.read_csv('Outputs/Sensitivity/Reliability_subsidy_sensitivity_analysis_tier' + str(i) + '.csv', index_col=0)
        tier_carbon = \
            pd.read_csv('Outputs/Sensitivity/' + 'Carbon_sensitivity_analysis_tier' + str(i) + '.csv')
        # tier_carbon = \
        #     pd.read_csv(SEAR.sensitivity_outputs + 'Carbon_sensitivity_analysis_tier' + str(i) + '.csv')
        off_grid_pv_rel = tier_reliability['pv_mini_grid_percentage_pop'] + tier_reliability['pv_SA_percentage_pop']
        off_grid_diesel_rel = \
            tier_reliability['diesel_mini_grid_percentage_pop'] + tier_reliability['diesel_SA_percentage_pop']
        off_grid_pv_carbon = tier_carbon['pv_mini_grid_percentage_pop'] + tier_carbon['pv_SA_percentage_pop']
        off_grid_diesel_carbon = \
            tier_carbon['diesel_mini_grid_percentage_pop'] + tier_carbon['diesel_SA_percentage_pop']

        ax0.plot(off_grid_pv_rel.index,off_grid_pv_rel, label='Off_grid_pv ' + str(i), linewidth=line_width
                 , color='#B1D665')
        ax0.plot(off_grid_diesel_rel.index, off_grid_diesel_rel, label='Off_grid_diesel ' + str(i),
                 linewidth=line_width, color='#FFA9AE') 
        ax0.plot(tier_reliability['grid_percentage_pop'].index, tier_reliability['grid_percentage_pop'], label='Grid ' + str(i),
                    linewidth=line_width, color='#9D8AC8')
        ax1.plot(off_grid_pv_carbon.index, off_grid_pv_carbon, label='Off_grid_pv ' + str(i), linewidth=line_width,
                 color='#B1D665')
        ax1.plot(off_grid_diesel_carbon.index,
                 off_grid_diesel_carbon, label='Off_grid_diesel ' + str(i), linewidth=line_width, color='#FFA9AE')
        ax1.plot(tier_carbon['grid_percentage_pop'].index, tier_carbon['grid_percentage_pop'], label='Grid ' + str(i),
                    linewidth=line_width, color='#9D8AC8')
        line_width += 1.5
    ax0.set_xlim(0, 1.0)
    ax1.set_xlim(1, 9)
    ax0.set_ylim(0, 0.7)
    ax1.set_ylim(0, 0.7)
    ax0.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_xticks(np.arange(1, 10, 1))
    ax1.set_xticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])
    # ax0.set_title('Reliability Sensitivity Analysis', fontsize=16)
    ax0.set_xlabel('Reliability penalty in 2022 $', fontsize=16)
    ax0.set_ylabel('Percentage of population of each Mode', fontsize=16)
    # ax1.set_title('Carbon Sensitivity Analysis', fontsize=16)
    ax1.set_xlabel('Percentile values of AR6 C1 & C2 Scenario Carbon Prices', fontsize=16)
    # plt.tight_layout()
    legend_elements = [Line2D([0], [0], color='black', lw=1.5, label='Tier 2'),
                          Line2D([0], [0], color='black', lw=3, label='Tier 3'),
                            Line2D([0], [0], color='black', lw=4.5, label='Tier 4'),
                       Patch(facecolor='#9D8AC8', label='Grid'),
                       Patch(facecolor='#B1D665', label='Off-grid PV'),
                       Patch(facecolor='#FFA9AE', label='Off-grid Diesel')]
    fig.legend(handles=legend_elements, loc='center', frameon=False, fontsize=14, bbox_to_anchor=(0.42, 0.78))
    # fig.legend(handles=legend_elements, loc='center', frameon=False, fontsize=14, bbox_to_anchor=(0.84, 0.38))
    plt.subplots_adjust(wspace=0.1, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.savefig('/Users/' + Plotting.home_directory + Plotting.plotting_output + 'Sensitivity_analysis.png', dpi=300)
    # if mode == 'subplot':
    #     # return the subplots as a tuple
    #     return ax0, ax1=

    plt.savefig('Outputs/Plotting_outputs/Sensitivity_analysis_penalty.pdf')
    plt.show()

# Function that plots the scenario results for the different modes on maps for each tier
def plot_subplot_maps_cat():

    # Set up parameters
    rel = 'full'
    scenario = 'CT_N_RP0.5'
    colors_4 = ["#D9D3DA", "#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB", "#F6CD41"]
    colors = ["#D9D3DA", "#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB"]
    colors_pies = ["#9D8AC8", "#B1D665", "#A7D2FB", "#FFA9AE", '#FFCA06']
    modes = ['grid', 'pv_mini_grid', 'pv_SA', 'diesel_mini_grid', 'diesel_SA']
    my_cmap = ListedColormap(colors, name="my_cmap")
    my_cmap_4 = ListedColormap(colors_4, name="my_cmap_4")
    
    # Import and merge map files
    map = gp.read_file('Geo/fishnet_with_countries_duplicates_removed.shp')
    stats_0 =  pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_1_' + scenario + '.csv')
    stats_1 =  pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_2_' + scenario + '.csv')
    stats_2 = pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_3_'+ scenario + '.csv')
    stats_3 = pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_4_' + scenario + '.csv')
    map_and_stats_0 = map.merge(stats_0, on='id')
    map_and_stats_1 = map.merge(stats_1, on='id')
    map_and_stats_2 = map.merge(stats_2, on= 'id')
    map_and_stats_3 = map.merge(stats_3, on= 'id')

    # Import summary information
    input_0 = pd.read_csv('Spatial/cost_optimal_output_summary_' + rel + '_universal_2030_tier_1_' + scenario + '.csv')
    input_1 = pd.read_csv('Spatial/cost_optimal_output_summary_' + rel + '_universal_2030_tier_2_' + scenario + '.csv')
    input_2 = pd.read_csv('Spatial/cost_optimal_output_summary_' + rel + '_universal_2030_tier_3_' + scenario + '.csv')
    input_3 = pd.read_csv('Spatial/cost_optimal_output_summary_' + rel + '_universal_2030_tier_4_' + scenario + '.csv')

    # Plot maps
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
    ax0 = map_and_stats_0.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax0, edgecolor=".4", rasterized=True)
    ax1 = map_and_stats_1.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax1, edgecolor=".4", rasterized=True)
    ax2 = map_and_stats_2.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax2, edgecolor=".4", rasterized=True)
    ax3 = map_and_stats_3.plot(column="mode", cmap=my_cmap_4, linewidth=0.0, ax=ax3, edgecolor=".4", rasterized=True)
    # bar_info = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=0, vmax=80000))
    # bar_info._A = []
    ax0.set_ylim(-4100000, 3300000)
    ax1.set_ylim(-4100000, 3300000)
    ax2.set_ylim(-4100000, 3300000)
    ax3.set_ylim(-4100000, 3300000)
    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax0.set_title('Tier 1', fontsize=16), ax1.set_title('Tier 2', fontsize=16), ax2.set_title('Tier 3', fontsize=16), ax3.set_title('Tier 4', fontsize=16)

    # Plot Figure legend
    nothing = mpatches.Patch(color='#8DBDA7', label='N/A')
    grid = mpatches.Patch(color='#9D8AC8', label='Grid')
    pv_mini = mpatches.Patch(color='#B1D665', label='PV Mini-grid')
    diesel_mini = mpatches.Patch(color='#FFA9AE', label='Diesel Mini-grid')
    pv_sa = mpatches.Patch(color='#A7D2FB', label='PV Stand Alone')
    diesel_sa = mpatches.Patch(color='#FFCA06', label='Diesel Stand Alone')
    # diesel_sa = mpatches.Patch(colour='', label='Diesel Stand Alone')
    plt.figlegend(handles=[nothing, grid, pv_mini, diesel_mini, pv_sa, diesel_sa], ncol=6, loc='lower center',
                  fontsize=16, frameon=False)

    # Plot pie charts
    ax7 = fig.add_axes([-0.005, 0.30, 0.13, 0.26])
    ax7.axis('off')
    x = []
    y = []
    labels = []
    for mode in modes:
        mode_column = input_0[mode + '_percentage_pop']
        if mode_column[0] == 0:
            pass
        else:
            y.append(mode_column[0] * 100)
            x.append(mode)
            label_int = int(round(mode_column[0] * 100, 0))
            labels.append(str(label_int) + '%')
    ax7 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # Set pie labels font size
    for text in ax7[1]:
        text.set_fontsize(12)
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax7 = plt.gcf()
    ax7.gca().add_artist(my_circle)

    # Plot pie charts
    ax4 = fig.add_axes([0.242, 0.30, 0.13, 0.26])
    ax4.axis('off')
    x = []
    y = []
    labels = []
    for mode in modes:
        mode_column = input_1[mode + '_percentage_pop']
        if mode_column[0] == 0:
            pass
        else:
            y.append(mode_column[0] * 100)
            x.append(mode)
            label_int = int(round(mode_column[0]*100, 0))
            labels.append(str(label_int) + '%')
    ax4 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    for text in ax4[1]:
        text.set_fontsize(12)
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax4 = plt.gcf()
    ax4.gca().add_artist(my_circle)

    ax5 = fig.add_axes([0.493, 0.30, 0.13, 0.26])
    ax5.axis('off')
    x = []
    y = []
    labels = []
    for mode in modes:
        mode_column = input_2[mode + '_percentage_pop']
        if mode_column[0] == 0:
            pass
        else:
            y.append(mode_column[0] * 100)
            x.append(mode)
            label_int = int(round(mode_column[0]*100, 0))
            labels.append(str(label_int) + '%')
    ax5 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    for text in ax5[1]:
        text.set_fontsize(12)
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax5 = plt.gcf()
    ax5.gca().add_artist(my_circle)
    ax6 = fig.add_axes([0.737, 0.30, 0.13, 0.26])
    ax6.axis('off')
    x = []
    y = []
    labels = []
    for mode in modes:
        mode_column = input_3[mode + '_percentage_pop']
        if mode_column[0] == 0:
            pass
        else:
            y.append(mode_column[0] * 100)
            x.append(mode)
            label_int = int(round(mode_column[0]*100, 0))
            labels.append(str(label_int) + '%')
    ax6 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    for text in ax6[1]:
        text.set_fontsize(12)
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax6 = plt.gcf()
    ax6.gca().add_artist(my_circle)
    plt.tight_layout()
    # plt.savefig('/Users/' + Plotting.home_directory +
    #             '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
    #             '/Outputs/Plotting_outputs/new_modes_universal_2030_' + rel + '_' + scenario + '.png', dpi=300)
    plt.savefig('Outputs/Plotting_outputs/new_modes_universal_2030_' + rel + '_' + scenario + '.pdf')
    plt.show()

# Function that plots the maps for the reliability sensitiviy analysis
def mapped_sensitivity_rel(tier):

    """
    Plotting function that prepares a map showing the penalty level at which there is a change in mode
    :param tier: energy access tier being looked at
    :return: A plot showing the price levels at which the mode changes
    """
    base_scenario = pd.read_csv('Spatial/cost_optimal_output_by_cell_full_universal_2030_tier_'+ str(tier) + '_CT_N_RP0.0.csv')
    base_scenario['mode_simplified'] = Utils().simplify_mode(base_scenario)
    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    reliability_penalties = np.arange(0.0, 1.05, 0.05)

    #Output df contains the mode at each penalty level for all, and separated for grid to pv, and grid to diesel
    output_df = pd.DataFrame()

    for penalty in reliability_penalties:
        # Ensure that penalty is two decimal places
        # penalty = round(penalty, 2)
        scenario_df = pd.read_csv('Outputs/Sensitivity/Reliability_penalty_sensitivity_analysis_'
                                        + str(penalty) +'_by_cell_tier_' + str(tier) + '.csv')
        scenario_df['mode_simplified'] = Utils().simplify_mode(scenario_df)
        scenario_df['mode_diff'] = scenario_df['mode_simplified'] - base_scenario['mode_simplified']
        scenario_df['grid_to_pv'] = Utils().isolate_layer(base_scenario, scenario_df, 1, -1)
        scenario_df['grid_to_diesel'] = Utils().isolate_layer(base_scenario, scenario_df, 1, -2)
        scenario_df['mode_diff'] = scenario_df['mode_diff'].replace(0, np.nan)

        # Filter out the values that are nan and just keep non nan values
        to_append = scenario_df[scenario_df['mode_diff'].notnull()]
        to_append['penalty'] = penalty

        # Append the values to the output df but check that the df does not already contain the values by the column 'id'
        if len(output_df) == 0:
            output_df = to_append
        else:
            # Filter out values that are not in 'id' of output df
            to_append = to_append[~to_append['id'].isin(output_df['id'])]
            output_df = pd.concat([output_df, to_append], axis=0)

    # Merge the output df with the fishnet layer
    map = gp.read_file('Geo/fishnet_with_countries_duplicates_removed.shp')
    map_and_stats = map.merge(output_df, on='id')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    africa_shp.plot(color='#D9D3DA', linewidth=0.0, ax=ax, edgecolor=".4")
    map_and_stats.plot(column="penalty", cmap='plasma', linewidth=0.0, ax=ax, edgecolor='0.8', legend=True, vmax=1, rasterized=True)
    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    # plt.savefig('Outputs/Plotting_outputs/reliability_mapped_sensitivity_tier_' + str(tier) + '.png', dpi=300)
    plt.savefig('Outputs/Plotting_outputs/reliability_mapped_sensitivity_tier_' + str(tier) + '.pdf')
    plt.tight_layout()
    plt.show()

# Function that plots the maps for the carbon price sensitiviy analysis
def mapped_sensitivity_carbon(tier):

    """
    Plotting function that prepares a map showing the carbon price level at which there is a change in mode
    :param tier: energy access tier being looked at
    :return: A plot showing the carbon price levels at which the mode changes
    """
    base_scenario = pd.read_csv('Spatial/cost_optimal_output_by_cell_full_universal_2030_tier_'+ str(tier) + '_CT_N_RP0.0.csv')
    base_scenario['mode_simplified'] = Utils().simplify_mode(base_scenario)
    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    carbon_prices = pd.read_csv(LEAF.sensitivity_inputs + 'carbon_prices.csv', index_col=0)

    #Output df contains the mode at each penalty level for all, and separated for grid to pv, and grid to diesel
    output_df = pd.DataFrame()

    for decile in carbon_prices.index:
        # Ensure that penalty is two decimal places
        # penalty = round(penalty, 2)
        scenario_df = pd.read_csv('Outputs/Sensitivity/Carbon_price_sensitivity_analysis_'
                                        + str(decile) +'_by_cell_tier_' + str(tier) + '.csv')
        scenario_df['mode_simplified'] = Utils().simplify_mode(scenario_df)
        scenario_df['mode_diff'] = scenario_df['mode_simplified'] - base_scenario['mode_simplified']
        # print(scenario_df['mode_diff'].value_counts())
        # scenario_df['grid_to_pv'] = Utils().isolate_layer(base_scenario, scenario_df, 1, -1)
        # scenario_df['grid_to_diesel'] = Utils().isolate_layer(base_scenario, scenario_df, 1, -2)
        scenario_df['mode_diff'] = scenario_df['mode_diff'].replace(0, np.nan)

        # Filter out the values that are nan and just keep non nan values
        to_append = scenario_df[scenario_df['mode_diff'].notnull()]
        to_append['price'] = decile

        # Append the values to the output df but check that the df does not already contain the values by the column 'id'
        if len(output_df) == 0:
            output_df = to_append
        else:
            # Filter out values that are not in 'id' of output df
            to_append = to_append[~to_append['id'].isin(output_df['id'])]
            output_df = pd.concat([output_df, to_append], axis=0)

    # Merge the output df with the fishnet layer
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    map_and_stats = map.merge(output_df, on='id')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    africa_shp.plot(color='#D9D3DA', linewidth=0.0, ax=ax, edgecolor=".4")
    map_and_stats.plot(column="price", cmap='plasma', linewidth=0.0, ax=ax, edgecolor='0.8', legend=True, rasterized=True)
    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    # plt.savefig('Outputs/Plotting_outputs/carbon_mapped_sensitivity_tier_' + str(tier) + '.png', dpi=300)
    plt.savefig('Outputs/Plotting_outputs/carbon_mapped_sensitivity_tier_' + str(tier) + '.pdf')
    plt.tight_layout()
    plt.show()

# Function that plots the maps for the investment requirement 
def investment_requirement_mapped(tier, scenario):

    single_mode = 'none' # 'grid', 'pv_investment_MG', 'diesel_investment_MG', 'pv_investment_SHS', 'diesel_sa', 'none'
    rel = 'full'
    per_cap = 'Y'
    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    base_layer = africa_shp.merge(LEAF.country_data_sheet, on='ISO3')
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    scenario_df = pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_'
        + str(tier) + '_' + str(scenario) + '.csv')
    population_gained_access = pd.read_csv('Spatial/No_access_pop/by_year_additions_'
                                           + LEAF.scenario + '_' + str(LEAF.target_year) + '.csv')['sum']
    households_gained_access = \
        pd.read_csv('Spatial/households_added_by_year_universal_2030.csv')['sum']
    households_gained_access = households_gained_access.replace(0, np.nan)
    household_installation_costs = pd.read_csv('Spatial/household_installation_costs_universal_2030.csv')['total']
    
    if single_mode != 'none':
        if single_mode == 'grid':
            scenario_df['investment_requirement'] = \
                pd.read_csv('Spatial/grid_extension_costs_' +
                            LEAF.scenario + '_' + str(LEAF.target_year) + '_tier_' + str(tier) + '.csv')['sum']
        else:
            scenario_df['investment_requirement'] = \
            pd.read_csv(SpatialData.spatial_filepath + 'off_grid/' + single_mode + '_' + rel + '_' + LEAF.scenario +
                        '_' + str(LEAF.target_year) + '_tier_' + str(tier) + '.csv')['sum']
    else:
        scenario_df['investment_requirement'] = scenario_df['total_investment']
    scenario_df['investment_requirement'] = scenario_df['investment_requirement'].replace(0, np.nan)
    scenario_df['investment_requirement'] = scenario_df['investment_requirement'] + household_installation_costs
    if per_cap == 'Y':
        scenario_df['investment_requirement'] = scenario_df['investment_requirement'] / households_gained_access

    else:
        pass
    # scenario_df['investment_requirement'] = scenario_df['investment_requirement'] + household_install_costs

    maps_and_stats = map.merge(scenario_df, on='id')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    base_layer.plot(color='#D9D3DA', linewidth=0.0, ax=ax, edgecolor=".4", rasterized=True)
    if per_cap == 'Y':
        maps_and_stats.plot(column="investment_requirement", cmap='plasma', linewidth=0.0, ax=ax, edgecolor='0.8',
                            legend=True, vmin=200)
    else:
        maps_and_stats.plot(column="investment_requirement",cmap='plasma', linewidth=0.0, ax=ax,
                            edgecolor='0.8', legend=True, vmax=3000000)

    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('Outputs/Plotting_outputs/investment_percap_' + per_cap + '_tier_' + str(tier) + '.pdf')
    plt.show()

# Function that makes a subplot of people requiring access to electricity by 2030
def simple_population_density_plot():

    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    map = gp.read_file('Geo/fishnet_with_countries_duplicates_removed.shp')
    base_layer = africa_shp.merge(LEAF.country_data_sheet, on='ISO3')
    # grid_lines = gp.read_file('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
    #                           '/DATA/SPATIAL/fishnet_planar/Geo/GRID_MAP_PLOTTING.shp')
    population_gained_access = pd.read_csv('Spatial/No_access_pop/by_year_additions_' + LEAF.scenario + '_' + str(LEAF.target_year) + '.csv')
    population_gained_access['summed_adjusted'] = population_gained_access['sum'].replace(0, np.nan)
    map_and_stats = map.merge(population_gained_access, on='id')
    fig, ax = plt.subplots(1, figsize=(8, 8))
    base_layer.plot(color='#D9D3DA', linewidth=0.1, ax=ax, edgecolor=".9")
    map_and_stats.plot(column="summed_adjusted", cmap='Reds', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=120000))
    sm._A = []
    # add vertical colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Population Requiring Access by 2030', fontsize=12)

    # grid_lines.plot(color='0.5', linewidth=0.1, ax=ax)
    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    # plt.rc('legend', fontsize=20)
    plt.savefig('Outputs/Plotting_outputs/population_requiring_access.pdf')
    plt.tight_layout()
    plt.show()

# Function that maps the emissions in each of the selected scenarios
def emissions_change_map(tier_of_access):

    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    map = gp.read_file('Geo/fishnet_with_countries_duplicates_removed.shp')
    baseline_emissions = pd.read_csv('Spatial/cost_optimal_output_by_cell_full_universal_2030_tier_'+ str(tier_of_access) + '_CT_N_RP0.0.csv')
    ctax_emissions = pd.read_csv('Spatial/cost_optimal_output_by_cell_full_universal_2030_tier_'+ str(tier_of_access) + '_CT_Y_RP0.0.csv')
    # population_gained_access = pd.read_csv('Spatial/No_access_pop/by_year_additions_'
    #                                        + LEAF.scenario + '_' + str(LEAF.target_year) + '.csv')
    ctax_emissions['diff'] = baseline_emissions['total_emissions'] - ctax_emissions['total_emissions']
    # ctax_emissions['diff'] = ctax_emissions['diff']/population_gained_access['sum']
    ctax_emissions['diff'] = ctax_emissions['diff'].replace(0, np.nan)
    ctax_emissions['diff'] = ctax_emissions['diff']/1000
    map_and_stats = map.merge(ctax_emissions, on='id')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    africa_shp.plot(color='#D9D3DA', linewidth=0.2, ax=ax, edgecolor=".8")
    map_and_stats.plot(column="diff", cmap='cool', linewidth=0.0, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=45000)
    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    # plt.savefig('Outputs/Plotting_outputs/emissions_change_CTAX_tier_' + str(tier_of_access)
    #             + '.png', dpi=300)
    plt.savefig('Outputs/Plotting_outputs/emissions_change_CTAX_tier_' + str(tier_of_access) + '.pdf')
    plt.tight_layout()
    plt.show()

# Plot single tier (3) map of difference from reference scenario to test scenario
def plot_scenario_shift_map(test_scenario, tier, rel):

    africa_shp = gp.read_file('Geo/Export_Output_africa_bound.shp')
    map = gp.read_file('Geo/fishnet_with_countries_duplicates_removed.shp')
    stats_reference =  pd.read_csv('Spatial/cost_optimal_output_by_cell_full_universal_2030_tier_'+ str(tier) + '_CT_N_RP0.0.csv')
    stats_scenario = pd.read_csv('Spatial/cost_optimal_output_by_cell_' + rel +'_universal_2030_tier_'+ str(tier) + '_' +
                                 str(test_scenario) + '.csv')
    base_layer = africa_shp.merge(LEAF.country_data_sheet, on='ISO3')
    population_gained_access = pd.read_csv('Spatial/No_access_pop/by_year_additions_'
                                           + LEAF.scenario + '_' + str(LEAF.target_year) + '.csv')['sum']
    stats_reference['mode_simplified'] = Utils().simplify_mode(stats_reference)
    stats_scenario['mode_simplified'] = Utils().simplify_mode(stats_scenario)
    stats_scenario['mode_diff'] = stats_reference['mode_simplified'] - stats_scenario['mode_simplified']

    # # Create individual layers on map
    stats_scenario['grid_to_pv'] = Utils().isolate_layer(stats_reference, stats_scenario, 1, -1)
    stats_scenario['grid_to_diesel'] = Utils().isolate_layer(stats_reference, stats_scenario, 1, -2)
    stats_scenario['pv_to_diesel'] = Utils().isolate_layer(stats_reference, stats_scenario, 2, -1)
    stats_scenario['pv_to_grid'] = Utils().isolate_layer(stats_reference, stats_scenario, 2, 1)
    stats_scenario['diesel_to_pv'] = Utils().isolate_layer(stats_reference, stats_scenario, 3, 1)
    stats_scenario['diesel_to_grid'] = Utils().isolate_layer(stats_reference, stats_scenario, 3, 2)

    # # Plot map layers
    map_and_stats = map.merge(stats_scenario, on='id')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    base_layer.plot(color='#D9D3DA', linewidth=0.0, ax=ax, edgecolor=".4", rasterized=True)
    # map_and_stats.plot(column="mode_diff", cmap='RdBu', linewidth=0.0, ax=ax, edgecolor='0.8', legend=False)
    # map_and_stats.plot(column="diesel_to_pv", cmap='winter_r', linewidth=0.0, ax=ax, edgecolor='0.8')
    map_and_stats.plot(column="diesel_to_pv", cmap='winter_r', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    map_and_stats.plot(column="diesel_to_grid", cmap='cool_r', linewidth=0.0, ax=ax, edgecolor='0.8',rasterized=True)
    map_and_stats.plot(column="grid_to_pv", cmap='winter', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    map_and_stats.plot(column="grid_to_diesel", cmap='summer_r', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    map_and_stats.plot(column="pv_to_diesel", cmap='Wistia_r', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    map_and_stats.plot(column="pv_to_grid", cmap='PiYG', linewidth=0.0, ax=ax, edgecolor='0.8', rasterized=True)
    # ax.set_title('Difference in mode choice between reference and test scenario')
    ax.set_ylim(-4100000, 3300000)
    ax.axis('off')
    #
    # Plot Figure legend
    # nothing = mpatches.Patch(color='#8DBDA7', label='N/A')
    diesel_to_pv = mpatches.Patch(color='#00FF7C', label='Diesel to PV')
    diesel_to_grid = mpatches.Patch(color='#FF00FA', label='Diesel to Grid')
    grid_to_pv = mpatches.Patch(color='#0700F8', label='Grid to PV')
    grid_to_diesel = mpatches.Patch(color='#FFFF69', label='Grid to Diesel')
    pv_to_diesel = mpatches.Patch(color='#FF7623', label='PV to Diesel')
    pv_to_grid = mpatches.Patch(color='#870048', label='PV to Grid')
    plt.figlegend(handles=[diesel_to_pv, diesel_to_grid, grid_to_pv, grid_to_diesel, pv_to_diesel, pv_to_grid], ncol=3,
                  loc='lower center',fontsize=14, frameon=False)

    # Add new axes in the bottom left of the figure
    ax2 = fig.add_axes([0.11, 0.35, 0.25, 0.19])
    # ax2.axis('off')
    mode_shifts = []
    labels = []
    modes = []
    # colours = ['#00FF7C', '#FF00FA', '#0700F8', '#FFFF69', '#FF7623', '#870048']
    # list_modes = ['diesel_to_pv', 'diesel_to_grid', 'grid_to_pv','grid_to_diesel', 'pv_to_diesel', 'pv_to_grid']
    colours = ['#0700F8', '#00FF7C', '#FF00FA']
    list_modes = ['grid_to_pv', 'diesel_to_pv','diesel_to_grid']
    # 
    if test_scenario == 'CT_Y_RP0.0':
        colours = ['#0700F8', '#00FF7C', '#FF00FA']
        list_modes = ['grid_to_pv', 'diesel_to_pv', 'diesel_to_grid']
    
    
    
    bar_colours = []
    for mode in range(0, len(list_modes)):
        mode_column = stats_scenario[list_modes[mode]]
        population_shift_mode = mode_column * population_gained_access
        summed_population_shift = population_shift_mode.sum()
        percentage_shift = summed_population_shift / population_gained_access.sum()
        percentage_shift = (percentage_shift * 100).round(0)
        modes.append(list_modes[mode])
        label = str(int(percentage_shift)) + '%'
        labels.append(label)
        mode_shifts.append(percentage_shift)
        bar_colours.append(colours[mode])
    # plot horizontal bar chart
    ax2 = plt.barh(modes, mode_shifts, align='center', color=bar_colours)
    ax2 = plt.yticks(modes, labels, fontsize=14)
    # set x axis limits
    ax2 = plt.xlim(0, 25)

    # Turn the x axes off
    ax2 = plt.gca().axes.get_xaxis().set_visible(False)

    # turn off all spines
    ax2 = plt.gca().spines['right'].set_visible(False)
    ax2 = plt.gca().spines['top'].set_visible(False)
    ax2 = plt.gca().spines['left'].set_visible(False)
    ax2 = plt.gca().spines['bottom'].set_visible(False)

    # set tick length to zero
    ax2 = plt.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    # plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A'
    #                        '/Outputs/Plotting_outputs/shift_map_' + test_scenario + '_tier_' + str(tier) + '.png', dpi=300)
    plt.savefig('Outputs/Plotting_outputs/shift_map_' + test_scenario + '_tier_' + str(tier) + '.pdf')


def plot_map():

    gnuplot = cm.get_cmap('gnuplot', 512)
    newcmp = ListedColormap(gnuplot(np.linspace(0.06, 1.0, 256)))
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A/Spatial'
                        '/cost_optimal_output_by_cell_full_universal_2030_tier_3_CT_N.csv')
    map_and_stats = map.merge(stats, on='id')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    map_and_stats.plot(column="total_emissions", cmap=newcmp, linewidth=0.0, ax=ax, edgecolor=".4")
    # overlay.plot()
    # plt.subplots_adjust(top=1, bottom=0, right=0.5, left=0,
    #                     hspace=0, wspace=0)
    # bar_info = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=120))
    bar_info = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=min(stats['total_emissions']),
                                                                     vmax=max(stats['total_emissions'])))
    bar_info._A = []
    cbar = fig.colorbar(bar_info, fraction=0.037)
    ax = ax.set_ylim(-4100000, 3300000)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # cbar = fig.colorbar(bar_info)
    # plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A'
    #                        '/Outputs/Plotting_outputs/2020_population_baseline_no_access_arial.png', dpi=300)
    plt.show()

def plot_map_divergant():

    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A/'
                        'Spatial/No_access_pop/baseline_annual_2030.csv')
    map_and_stats = map.merge(stats, on='id')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    map_and_stats.plot(column="diff", cmap='coolwarm', linewidth=0.0, ax=ax, edgecolor=".4", vmin=-10000, vmax=10000)
    # overlay.plot()
    # plt.subplots_adjust(top=1, bottom=0, right=0.5, left=0,
    #                     hspace=0, wspace=0)
    # bar_info = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=120))
    bar_info = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-10000, vmax=10000))
    bar_info._A = []
    ax.set_title('Change in Population without Access 2020-30')
    cbar = fig.colorbar(bar_info, fraction=0.032)
    ax = ax.set_ylim(-4100000, 3300000)
    # plt.colorbar(fig)
    # plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    plt.axis('off')

    # cbar = fig.colorbar(bar_info)
    plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A'
                           '/Outputs/Plotting_outputs/2030_population_baseline_no_access_diverging_arial.png', dpi=300)
    plt.show()

def plot_map_categories():

    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A'
                        '/Spatial/cost_optimal_output_by_cell_full_universal_2035_tier_2.csv')
    map_and_stats = map.merge(stats, on='id')
    # ax = map_and_stats.plot(column='mode', categorical=True, legend=True,
    #                         legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1, 0.5),'fmt': "{:.0f}"})

    # plt.rcParams.update({
    #     "font.weight": "light"
    # })

    fig, ax = plt.subplots(1, figsize=(10, 10)) #figsize=(15, 15)
    map_and_stats.plot(column="mode", cmap='Set2', linewidth=0.0, ax=ax, edgecolor=".4")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    # bar_info = plt.cm.ScalarMappable(cmap="Spectral")
    # bar_info._A = []
    # cbar = fig.colorbar(bar_info)
    ax = ax.set_ylim(-4100000, 3300000)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    #leg.set_bbox_to_anchor((1.15, 0.5))

    # cmap = cm.get_cmap('tab10', 5)  # PiYG
    #
    # for i in range(cmap.N):
    #     rgba = cmap(i)
    #     # rgb2hex accepts rgb or rgba
    #     print(matplotlib.colors.rgb2hex(rgba))

    nothing = mpatches.Patch(color='#7EBFA6', label='N/A')
    grid = mpatches.Patch(color='#919FC8', label='Grid')
    pv_mini = mpatches.Patch(color='#B1D665', label='PV Mini-grid')
    diesel_mini = mpatches.Patch(color='#DFC699', label='Diesel Mini-grid')
    pv_sa = mpatches.Patch(color='#B3A5B3', label='PV Stand Alone')
    # diesel_sa = mpatches.Patch(colour='', label='Diesel Stand Alone')
    plt.figlegend(handles=[nothing, grid, pv_mini, diesel_mini, pv_sa], ncol=5, loc='lower center',
                  fontsize='x-large', frameon=False)
    plt.tight_layout()
    plt.savefig('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/LEAF_A'
                '/Outputs/Plotting_outputs/mode_universal_2035_full_CT0_tier_3.png', dpi=300)
    plt.show()

def bar_plots():

    sns.set_style=('darkgrid')
    # colors = ["#8DBDA7", "#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB"]
    # modes = ['pv_mini_grid', 'diesel_mini_grid', 'diesel_SA', 'pv_SA', 'grid']
    my_palette = {'pv_mini_grid': "#B1D665", 'diesel_mini_grid': "#FFA9AE", 'pv_SA': '#A7D2FB', 'diesel_SA': "#FFA9AE",
                  'grid': '#9D8AC8'}
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Generate some sequential data
    input_csv_1 = pd.read_csv(SpatialData. spatial_filepath +
                              'cost_optimal_output_summary_full_universal_2035_tier_1_CT_N.csv')
    x = SEAR.modes
    y1 = []
    for mode in SEAR.modes:
        mode_column = input_csv_1[mode + '_percentage_pop']
        y1.append(mode_column[0])
    sns.barplot(x=x, y=y1, palette=my_palette, ax=ax1, width=0.25)
    ax1.set_ylabel("% of population")

    # Center the data to make it diverging
    input_csv_2 = pd.read_csv(SpatialData. spatial_filepath +
                              'cost_optimal_output_summary_full_universal_2035_tier_2_CT_N.csv')
    y2 = []
    for mode in SEAR.modes:
        mode_column = input_csv_2[mode + '_percentage_pop']
        y2.append(mode_column[0])
    sns.barplot(x=x, y=y2, palette=my_palette, ax=ax2)

    input_csv_3 = pd.read_csv(SpatialData. spatial_filepath +
                              'cost_optimal_output_summary_full_universal_2035_tier_3_CT_N.csv')
    y3 = []
    for mode in SEAR.modes:
        mode_column = input_csv_3[mode + '_percentage_pop']
        y3.append(mode_column[0])
    sns.barplot(x=x, y=y3, palette=my_palette, ax=ax3)

    # sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    # plt.tight_layout(h_pad=2)
    # plt.xticks(rotation=90)
    plt.show()





# Function that joins csv values with fishnet and exports to shp file for plotting in QGIS.
def join_attribute_centroids(input_csv, input_filename):

    fishnet_geopandas = gp.read_file('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
                                     '/DATA/SPATIAL/fishnet/Grid/centroids_fuller_join_fids.shp')
    output = fishnet_geopandas.merge(input_csv, on='fid')
    output.to_file(FishNet.plotting_output_filepath + input_filename + ".shp")
    print('File ' + input_filename + ' merged and saved successfully to ' + FishNet.plotting_output_filepath)


# Function that joins csv values with fishnet and exports to shp file for plotting in QGIS.
def join_attribute(input_csv, input_filename):

    fishnet_geopandas = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries.shp')
    output = fishnet_geopandas.merge(input_csv, on='id')
    output.to_file(FishNet.plotting_output_filepath + input_filename + ".shp")
    print('File ' + input_filename + ' merged and saved successfully to ' + FishNet.plotting_output_filepath)


# Sort fid data by country and output as table of the summed values of all variables by country listed
def arrange_values_by_country(input_fid_file):

    header_list = input_fid_file.columns.values.tolist()
    header_list.remove('fid')
    header_list.remove('ISO3')


if __name__ == "__main__":
    main()




