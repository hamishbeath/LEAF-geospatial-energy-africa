from main import *
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import geopandas as gp
import pylab as pl
from matplotlib import rcParams
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import squarify
# from main_planar import *

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


class FishNet:
    geo_file_path = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/DATA/SPATIAL/fishnet_planar/Geo/'
    plotting_output_filepath = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/' \
                               'Outputs/plotting_materials/'

class Plotting:

    home_directory = 'hamishbeath'  # home_directory = 'hrb16
    plotting_output = '~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/' \
                               'Outputs/plotting_materials/'
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
    # bubble_plot()
    # bar_plots()
    make_stacked_bars_scenarios()
    # plot_heatmaps_with_icons()
    # plot_sensitivity()
    # plot_africa_choropleth()
    # plot_africa_choropleth_shares_diff(3)
    # plot_africa_choropleth_shares_PV()

def plot_africa_choropleth():

    indicator = 'diesel'    # options: 'diesel', 'Rural_energy_met_dec', 'Grid Emissions Intensity (kgCO2/kWh),
                    # 'demand_growth_factor', 'grid_density

    #Import Africa shapefile:
    africa_shp = gp.read_file('~/Library/Mobile Documents/com~apple~CloudDocs'
                              '/Mitigation_project/DATA/SPATIAL/fishnet_planar/Geo/PLOTTING_Export_Output_africa_bound.shp')
    africa_stats = SEAR.country_data_sheet
    map_and_stats = africa_shp.merge(africa_stats, on='ISO3')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(36, 7))
    ax1 = map_and_stats.plot(column="diesel_sub", cmap='hot_r', linewidth=0.5, ax=ax1, edgecolor=".4", legend=True)
    ax2 = map_and_stats.plot(column="Rural_energy_met_dec", cmap='hot', linewidth=0.5, ax=ax2, edgecolor=".4", legend=True, vmin=0.3)
    ax3 = map_and_stats.plot(column="Grid Emissions Intensity (kgCO2/kWh)", cmap='hot_r', linewidth=0.5, ax=ax3, edgecolor=".4",legend=True)
    ax4 = map_and_stats.plot(column="demand_growth_factor", cmap='hot_r', linewidth=0.5, ax=ax4, edgecolor=".4", legend=True) #legend=True, #legend_loc='lower center')
    # ax5 = map_and_stats.plot(column="grid_density", cmap='hot', linewidth=0.5, ax=ax5, edgecolor=".4", legend=True)
    ax1.set_title('Diesel Prices (2022 USD/litre)')
    ax2.set_title('National Grid Est. % Rural Energy Met')
    ax3.set_title('Grid Emissions Intensity (kgCO2/kWh')
    ax4.set_title('Grid Demand Growth Factor')
    # ax5.set_title('Population-weighted mean distance from grid (km)')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    # ax5.axis('off')
    # place all legends from each axis at bottom of subplots
    plt.savefig('/Users/'
                + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                                            '/Outputs/Plotting_outputs/' + 'heatmap_country_indicators.png')
    plt.tight_layout()
    plt.show()

def plot_africa_choropleth_shares_diff(tier_of_access):

    winter = cm.get_cmap('winter', 512)
    newcmp = ListedColormap(winter(np.linspace(0.30, 1.00, 256)))
    #Import Africa shapefile:
    africa_shp = gp.read_file('~/Library/Mobile Documents/com~apple~CloudDocs'
                              '/Mitigation_project/DATA/SPATIAL/fishnet_planar/Geo/PLOTTING_Export_Output_africa_bound.shp')
    africa_stats = pd.read_csv(Plotting.plotting_output + 'heatmap_diff_out_tier_' + str(tier_of_access) + '.csv')
    map_and_stats = africa_shp.merge(africa_stats, on='ISO3')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    base_layer = africa_shp.merge(SEAR.country_data_sheet, on='ISO3')
    ax1 = base_layer.plot(color='#BAB3BA', linewidth=0.5, ax=ax1, edgecolor=".4")
    ax2 = base_layer.plot(color='#BAB3BA', linewidth=0.5, ax=ax2, edgecolor=".4")
    ax1 = map_and_stats.plot(column="CT_N_RP0.5", cmap=newcmp, linewidth=0.5, ax=ax1, edgecolor=".4", legend=True, vmin=-0.0, vmax=0.45)
    ax2 = map_and_stats.plot(column="CT_Y_RP0.0", cmap=newcmp, linewidth=0.5, ax=ax2, edgecolor=".4", legend=True, vmin=-0.0, vmax=0.45)

    # ax5 = map_and_stats.plot(column="grid_density", cmap='hot', linewidth=0.5, ax=ax5, edgecolor=".4", legend=True)
    ax1.set_title('Difference in share of off-grid PV with $0.50/kWh Unmet Demand Penalty')
    ax2.set_title('Difference in share of off-grid PV with Carbon Price Scheme Applied')

    # ax5.set_title('Population-weighted mean distance from grid (km)')
    ax1.axis('off')
    ax2.axis('off')

    # ax5.axis('off')
    # place all legends from each axis at bottom of subplots
    plt.savefig('/Users/'
                + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                                            '/Outputs/Plotting_outputs/' + 'heatmap_diff_out_tier_' +
                str(tier_of_access) + '.png')
    plt.tight_layout()

    plt.show()

def plot_africa_choropleth_shares_PV():

    # winter = cm.get_cmap('winter', 512)
    # newcmp = ListedColormap(winter(np.linspace(0.30, 1.00, 256)))
    #Import Africa shapefile:
    africa_shp = gp.read_file('~/Library/Mobile Documents/com~apple~CloudDocs'
                              '/Mitigation_project/DATA/SPATIAL/fishnet_planar/Geo/PLOTTING_Export_Output_africa_bound.shp')
    africa_stats = pd.read_csv('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
                               '/SEAR_A/Spatial/cost_optimal_output_countries_summary_full_universal_2030_'
                               'tier_3_CT_N_RP0.0.csv')
    africa_grid_map = gp.read_file('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project'
                                   '/DATA/SPATIAL/fishnet_planar/Geo/GRID_MAP_PLOTTING.shp')
    africa_stats['PV_total'] = africa_stats['pv_SA_percentage_pop'] + africa_stats['pv_mini_grid_percentage_pop']
    map_and_stats = africa_shp.merge(africa_stats, on='ISO3')
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    base_layer = africa_shp.merge(SEAR.country_data_sheet, on='ISO3')
    ax1 = base_layer.plot(color='#BAB3BA', linewidth=0.5, ax=ax1, edgecolor=".4")
    ax1 = map_and_stats.plot(column="PV_total", cmap='summer_r', linewidth=0.5, ax=ax1, edgecolor=".4", legend=True, vmin=0, vmax=0.9)
    ax1 = africa_grid_map.plot(color='000000', linewidth=0.15, ax=ax1, edgecolor=".4")

    # ax5 = map_and_stats.plot(column="grid_density", cmap='hot', linewidth=0.5, ax=ax5, edgecolor=".4", legend=True)
    # ax1.set_title('Difference in share of off-grid PV with $0.50/kWh Unmet Demand Penalty')
    # ax2.set_title('Difference in share of off-grid PV with Carbon Price Scheme Applied')

    # ax5.set_title('Population-weighted mean distance from grid (km)')
    ax1.axis('off')

    # ax5.axis('off')
    # place all legends from each axis at bottom of subplots
    plt.savefig('/Users/'
                + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                                            '/Outputs/Plotting_outputs/' + 'heatmap_with_grid_out_tier_3.png')
    plt.tight_layout()

    plt.show()

def plot_sensitivity():
    # plt.style.use('fivethirtyeight')
    plt.style.use('seaborn')
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(17, 7.5))
    # set seaborn theme
    sns.set_theme(style="whitegrid")
    line_width = 1.0
    for i in range(2, 5):
        tier_reliability = \
            pd.read_csv(SEAR.sensitivity_outputs +  'Reliability_penalty_sensitivity_analysis_tier' + str(i) + '.csv', index_col=0)
        tier_carbon = \
            pd.read_csv(SEAR.sensitivity_outputs + 'Carbon_sensitivity_analysis_tier' + str(i) + '.csv')
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
        line_width += 1.0
    ax0.set_xlim(0, 1.0)
    ax1.set_xlim(1, 9)
    ax0.set_ylim(0, 0.7)
    ax1.set_ylim(0, 0.7)
    ax0.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_xticks(np.arange(1, 10, 1))
    ax1.set_xticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])
    ax0.set_title('Reliability Sensitivity Analysis', fontsize=16)
    ax0.set_xlabel('Reliability penalty in 2022 $', fontsize=16)
    ax0.set_ylabel('Percentage of population of each Mode', fontsize=16)
    ax1.set_title('Carbon Sensitivity Analysis', fontsize=16)
    ax1.set_xlabel('Percentile values of AR6 C1 & C2 Scenario Carbon Prices', fontsize=16)
    # plt.tight_layout()
    legend_elements = [Line2D([0], [0], color='black', lw=1, label='Tier 2'),
                          Line2D([0], [0], color='black', lw=2, label='Tier 3'),
                            Line2D([0], [0], color='black', lw=3, label='Tier 4'),
                       Patch(facecolor='#9D8AC8', label='Grid'),
                       Patch(facecolor='#B1D665', label='Off-grid PV'),
                       Patch(facecolor='#FFA9AE', label='Off-grid Diesel')]
    fig.legend(handles=legend_elements, loc='center', frameon=False, fontsize=14, bbox_to_anchor=(0.42, 0.78))
    plt.subplots_adjust(wspace=0.1, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.savefig('/Users/' + Plotting.home_directory + Plotting.plotting_output + 'Sensitivity_analysis.png', dpi=300)
    plt.savefig('/Users/' + Plotting.home_directory +
                '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                '/Outputs/Plotting_outputs/Sensitivity_analysis.png', dpi=300)
    plt.show()

# Function that makes stacked bars but for single tiers multiple scenarios
def make_stacked_bars_scenarios():

    fig_size = (14, 7)
    tier_of_access = 3
    f = plt.figure(figsize=fig_size)
    plt.axis('off')
    scenarios = ['_CT_N_RP0.0', '_CT_N_RP0.5', '_CT_Y_RP0.0']
    num = 1
    for scenario in scenarios:
        f = stacked_horizontal_bars(tier_of_access, f, scenario, num)
        num += 1
    plt.savefig('/Users/' + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                                              '/Outputs/Plotting_outputs/country_stacked_bars_scenarios_tier_' + str(tier_of_access) + '.png',dpi=300)

    plt.show()

def make_stacked_bars():

    # For loop function that calls stacked_horizontal_bars() and makes subplots for each tier 1-4
    # fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    fig_size = (15, 7.5)
    f = plt.figure(figsize=fig_size)
    # title = 'By-country proportion of population met by each share\n'
    # plt.title(title, fontsize=24)
    plt.axis('off')
    scenario = '_CT_N_RP0.0'
    num = 1
    for i in range(2, 5):
        f = stacked_horizontal_bars(i, f, scenario, num)
        num += 1
        # ax_loop = stacked_horizontal_bars(i + 1)
        # ax = ax_loop
    # ax1 = stacked_horizontal_bars(1)
    # ax2 = stacked_horizontal_bars(2)
    # ax3 = stacked_horizontal_bars(3)
    # ax4 = stacked_horizontal_bars(4)
    # plt.tight_layout()
    plt.savefig('/Users/' + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Outputs/Plotting_outputs/country_stacked_bars_' + scenario + '.png', dpi=300)
    plt.show()

def stacked_horizontal_bars(tier, f, scenario, num):

    tier = str(tier)
    input_df = pd.read_csv(SpatialData.spatial_filepath + 'cost_optimal_output_countries_summary_full_universal_2030_'
                                                          'tier_' + tier + scenario + '.csv')
    # input_df['total_off_grid_pv_share'] = input_df['pv_mini_grid_percentage_pop'] + input_df['pv_SA_percentage_pop']
    input_df = input_df.sort_values(by='Country', axis=0, ascending=False)
    input_df = input_df.loc[input_df['total_population'] > 0]
    input_df = input_df.set_index('Country')
    colors = ["#B1D665", "#A7D2FB", "#FFA9AE", '#FFCA06', "#9D8AC8"]
    labels = ['PV Mini-grid', 'PV Stand-alone', 'Diesel Mini-grid', 'Diesel Stand-alone','Grid']
    fields = ['pv_mini_grid_percentage_pop', 'pv_SA_percentage_pop', 'diesel_mini_grid_percentage_pop', 'diesel_SA_percentage_pop', 'grid_percentage_pop']
    # subtitle = 'Tier ' + tier
    subtitle = ''
    # fig, ax = plt.subplots(1, figsize=(15, 10))
    axarr = f.add_subplot(1, 3, num)
    left = len(input_df) * [0]
    for idx, name in enumerate(fields):
        plt.barh(input_df.index, input_df[name], left = left, color=colors[idx]) #colors=colors)
        left = left + input_df[name]
    # title and subtitle
    # plt.title(title, loc='left')
    plt.text(0, axarr.get_yticks()[-1] + 0.75, subtitle)
    # legend
    if num == 3:
        plt.legend(labels, bbox_to_anchor=([-0.10, -0.06, 0, 0]), ncol=5, frameon=False)
    # y axis name labels
    if num > 1:
        axarr.yaxis.set_ticklabels([])

    # remove spines
    axarr.spines['right'].set_visible(False)
    axarr.spines['left'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.spines['bottom'].set_visible(False)
    # format x ticks
    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.xticks(xticks, xlabels)
    # adjust limits and draw grid lines
    plt.ylim(-0.5, axarr.get_yticks()[-1] + 0.5)
    axarr.xaxis.grid(color='gray', linestyle='dashed')
    # print(type(ax))
    # print(type(plt))

    return f
    # plt.show()

def bubble_plot():

    sns.set_style('darkgrid')
    # Load the example mpg dataset
    input = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Outputs'
                        '/plotting_materials/bubble_plot_inputs_no_gaps.csv')

    # Plot miles per gallon against horsepower with other semantics
    sns.relplot(x="GDP/cap 2030", y="rate 2030", hue="region", size="population 2030",
                 alpha=.5,height=6, palette='bright',data=input)
    plt.show()

def bubble_plotly():

    input = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Outputs'
                        '/plotting_materials/bubble_plot_inputs_no_gaps.csv')

    fig = px.scatter(input, x="GDP/cap 2030", y="rate 2030", size="population 2030", color="region",
                     text="Country", size_max=150, log_x=True)
    fig.show()

def plot_subplot_maps():


    # gnuplot = cm.get_cmap('gnuplot2', 512)
    # newcmp = ListedColormap(gnuplot(np.linspace(0.15, 1.00, 256)))
    # edit the color map to have grey for 0 values


    # gnuplot = cm.get_cmap('hot_r', 512)
    # newcmp = ListedColormap(gnuplot(np.linspace(0.06, 1.0, 256)))
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats =  pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/'
                        'Spatial/No_access_pop/baseline_annual_2030.csv')
    map_and_stats = map.merge(stats, on='id')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))
    ax1 = map_and_stats.plot(column="2020", cmap='nipy_spectral_r', linewidth=0.0, ax=ax1, edgecolor=".4", vmin=0, vmax=80000)
    ax2 = map_and_stats.plot(column="2030", cmap='nipy_spectral_r', linewidth=0.0, ax=ax2, edgecolor=".4", vmin=0, vmax=80000)
    bar_info = plt.cm.ScalarMappable(cmap='nipy_spectral_r', norm=plt.Normalize(vmin=0, vmax=80000))
    bar_info._A = []
    ax1.set_ylim(-4100000, 3300000)
    ax2.set_ylim(-4100000, 3300000)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('2020')
    ax2.set_title('2030')
    cax = fig.add_axes([ax2.get_position().x1 + 0.02, ax2.get_position().y0, 0.02, ax2.get_position().height])
    plt.colorbar(bar_info, cax=cax)
    plt.subplots_adjust(wspace=0.01, left=-0.01)
    plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Outputs/Plotting_outputs/2020_2030_no_access_arial.png', dpi=300)
    plt.show()

def plot_subplot_maps_three_cat():

    # Set up parameters
    rel = 'full'
    scenario = 'CT_Y_RP0.0'
    colors = ["#8DBDA7", "#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB"]
    colors_pies = ["#9D8AC8", "#B1D665", "#A7D2FB", "#FFA9AE", '#FFCA06']
    modes = ['grid', 'pv_mini_grid', 'pv_SA', 'diesel_mini_grid', 'diesel_SA']
    my_cmap = ListedColormap(colors, name="my_cmap")

    # Import and merge map files
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats_0 =  pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_1_' + scenario + '.csv')
    stats_1 =  pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Spatial/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_2_' + scenario + '.csv')
    stats_2 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_3_'+ scenario + '.csv')
    stats_3 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_by_cell_' + rel + '_universal_2030_tier_4_' + scenario + '.csv')
    map_and_stats_0 = map.merge(stats_0, on='id')
    map_and_stats_1 = map.merge(stats_1, on='id')
    map_and_stats_2 = map.merge(stats_2, on= 'id')
    map_and_stats_3 = map.merge(stats_3, on= 'id')

    # Import summary information
    input_0 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_' + rel + '_universal_2030_tier_1_' + scenario + '.csv')
    input_1 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_' + rel + '_universal_2030_tier_2_' + scenario + '.csv')
    input_2 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_' + rel + '_universal_2030_tier_3_' + scenario + '.csv')
    input_3 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_' + rel + '_universal_2030_tier_4_' + scenario + '.csv')

    # Plot maps
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
    ax0 = map_and_stats_0.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax0, edgecolor=".4")
    ax1 = map_and_stats_1.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax1, edgecolor=".4")
    ax2 = map_and_stats_2.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax2, edgecolor=".4")
    ax3 = map_and_stats_3.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax3, edgecolor=".4")
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
    plt.savefig('/Users/' + Plotting.home_directory +
                '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                '/Outputs/Plotting_outputs/modes_universal_2030_' + rel + '_' + scenario + '.png', dpi=300)
    plt.show()

def plot_subplot_maps_four_cat():

    # Set up parameters
    rel = 'grid'
    scenario = 'CT_N_RP0'
    colors = ["#8DBDA7", "#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB"]
    colors_pies = ["#9D8AC8", "#B1D665", "#FFA9AE", "#A7D2FB"]
    modes = ['grid', 'pv_mini_grid', 'diesel_mini_grid', 'pv_SA']
    my_cmap = ListedColormap(colors, name="my_cmap")

    # Import and merge map files
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats_1 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                          '/Spatial/cost_optimal_output_by_cell_'+ rel +'_universal_2030_tier_1_' + scenario + '.csv')
    stats_2 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_by_cell_'+ rel +'_universal_2030_tier_2_' + scenario + '.csv')
    stats_3 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_by_cell_'+ rel +'_universal_2030_tier_3_' + scenario + '.csv')
    stats_4 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_by_cell_'+ rel +'_universal_2030_tier_4_' + scenario + '.csv')
    map_and_stats_1 = map.merge(stats_1, on='id')
    map_and_stats_2 = map.merge(stats_2, on='id')
    map_and_stats_3 = map.merge(stats_3, on='id')
    map_and_stats_4 = map.merge(stats_4, on='id')
    # Import summary information
    input_1 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_'+ rel +'_universal_2030_tier_1_' + scenario + '.csv')
    input_2 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_'+ rel +'_universal_2030_tier_2_' + scenario + '.csv')
    input_3 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_'+ rel +'_universal_2030_tier_3_' + scenario + '.csv')
    input_4 = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
                          '/cost_optimal_output_summary_'+ rel +'_universal_2030_tier_4_' + scenario + '.csv')
    # Plot maps
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
    ax1 = map_and_stats_1.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax1, edgecolor=".4")
    ax2 = map_and_stats_2.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax2, edgecolor=".4")
    ax3 = map_and_stats_3.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax3, edgecolor=".4")
    ax4 = map_and_stats_4.plot(column="mode", cmap=my_cmap, linewidth=0.0, ax=ax3, edgecolor=".4")
    # bar_info = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=0, vmax=80000))
    # bar_info._A = []
    ax1.set_ylim(-4100000, 3300000)
    ax2.set_ylim(-4100000, 3300000)
    ax3.set_ylim(-4100000, 3300000)
    ax4.set_ylim(-4100000, 3300000)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax1.set_title('Tier 1', fontsize=16)
    ax2.set_title('Tier 2', fontsize=16)
    ax3.set_title('Tier 3', fontsize=16)
    ax4.set_title('Tier 4', fontsize=16)

    # Plot Figure legend
    nothing = mpatches.Patch(color='#8DBDA7', label='N/A')
    grid = mpatches.Patch(color='#9D8AC8', label='Grid')
    pv_mini = mpatches.Patch(color='#B1D665', label='PV Mini-grid')
    diesel_mini = mpatches.Patch(color='#FFA9AE', label='Diesel Mini-grid')
    pv_sa = mpatches.Patch(color='#A7D2FB', label='PV Stand Alone')
    # diesel_sa = mpatches.Patch(colour='', label='Diesel Stand Alone')
    plt.figlegend(handles=[nothing, grid, pv_mini, diesel_mini, pv_sa], ncol=5, loc='lower center',
                  fontsize=16, frameon=False)

    # Plot pie charts
    ax5 = fig.add_axes([0.015, 0.30, 0.13, 0.26])
    ax5.axis('off')
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
            label_int = int(mode_column[0] * 100)
            labels.append(str(label_int) + '%')
    ax5 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax5 = plt.gcf()
    ax5.gca().add_artist(my_circle)

    ax6 = fig.add_axes([0.25, 0.30, 0.13, 0.26])
    ax6.axis('off')
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
            label_int = int(mode_column[0] * 100)
            labels.append(str(label_int) + '%')
    ax6 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax6 = plt.gcf()
    ax6.gca().add_artist(my_circle)

    ax7 = fig.add_axes([0.55, 0.30, 0.13, 0.26])
    ax7.axis('off')
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
            label_int = int(mode_column[0] * 100)
            labels.append(str(label_int) + '%')
    ax7 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax7 = plt.gcf()
    ax7.gca().add_artist(my_circle)

    ax8 = fig.add_axes([0.75, 0.30, 0.13, 0.26])
    ax8.axis('off')
    x = []
    y = []
    labels = []
    for mode in modes:
        mode_column = input_4[mode + '_percentage_pop']
        if mode_column[0] == 0:
            pass
        else:
            y.append(mode_column[0] * 100)
            x.append(mode)
            label_int = int(mode_column[0] * 100)
            labels.append(str(label_int) + '%')
    ax8 = plt.pie(y, colors=colors_pies, labels=labels, startangle=90)
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.5, color='white')
    ax8 = plt.gcf()
    ax8.gca().add_artist(my_circle)

    # Export plot
    # plt.tight_layout()
    # plt.savefig('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
    #             '/Outputs/Plotting_outputs/modes_4_universal_2030_'+ rel +'_' + scenario + '.png', dpi=300)
    plt.show()

def plot_map():

    gnuplot = cm.get_cmap('gnuplot', 512)
    newcmp = ListedColormap(gnuplot(np.linspace(0.06, 1.0, 256)))
    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/Spatial'
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
    # plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
    #                        '/Outputs/Plotting_outputs/2020_population_baseline_no_access_arial.png', dpi=300)
    plt.show()

def plot_map_divergant():

    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A/'
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
    plt.savefig('/Users/hamishbeath/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Outputs/Plotting_outputs/2030_population_baseline_no_access_diverging_arial.png', dpi=300)
    plt.show()

def plot_map_categories():

    map = gp.read_file(FishNet.geo_file_path + 'fishnet_with_countries_duplicates_removed.shp')
    stats = pd.read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
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
    plt.savefig('/Users/hrb16/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
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


def plot_heatmaps_with_icons():

    scenarios = ['CT_N_RP0.0',  'CT_N_RP0.5', 'CT_Y_RP0.0']
    averages = pd.DataFrame()
    winter = cm.get_cmap('winter', 512)
    newcmp = ListedColormap(winter(np.linspace(0.20, 1.00, 256)))

    # Loop over tiers of access
    for i in range(1, 5):

        heatmap_df_out = pd.DataFrame()
        heatmap_diff_out = pd.DataFrame()
        # Loop over scenarios
        for scenario in scenarios:

            # Read in the data
            input_csv = \
                pd.read_csv(SpatialData. spatial_filepath + 'cost_optimal_output_countries_summary_full_universal_2030_tier_'
                            + str(i) + '_' + scenario + '.csv')
            input_csv = input_csv.loc[input_csv['total_population'] > 0]
            heatmap_df_out['countries'] = input_csv['Country']
            heatmap_diff_out['countries'] = input_csv['Country']
            heatmap_diff_out['ISO3'] = input_csv['ISO3']
            heatmap_df_out[scenario] = \
                input_csv['pv_SA_percentage_pop'] + input_csv['pv_mini_grid_percentage_pop']
            averages[scenario] = heatmap_df_out[scenario]
            if scenario == 'CT_N_RP0.0':
                heatmap_diff_out[scenario] = heatmap_df_out[scenario] - heatmap_df_out[scenario]
            else:
                heatmap_diff_out[scenario] = heatmap_df_out[scenario] - heatmap_df_out['CT_N_RP0.0']

        # Output heatmap values
        heatmap_df_out.set_index('countries', inplace=True)
        heatmap_diff_out.set_index('countries', inplace=True)
        print(heatmap_df_out)
        heatmap_df_out.to_csv(Plotting. plotting_output + 'heatmap_df_out_tier_' + str(i) + '.csv')
        heatmap_diff_out.to_csv(Plotting. plotting_output + 'heatmap_diff_out_tier_' + str(i) + '.csv')
    averages['ISO3'] = heatmap_df_out.index
    averages.set_index('ISO3', inplace=True)


    # Plot heatmaps
    fig_size = (10, 8)
    f = plt.figure(figsize=fig_size)
    plt.tight_layout()
    # title = 'By-country proportion of population met by each share\n'
    # plt.title(title, fontsize=24)
    plt.axis('off')
    # scenario = '_CT_Y_RP0'
    num = 1
    for tier in range(2,5):

        f = make_heatmap(f, tier, num, newcmp)
        num += 1
    # plt.tight_layout()
    # bar_info = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=0, vmax=1))
    # bar_info._A = []
    # plt.colorbar(bar_info)
    plt.savefig('/Users/' + Plotting.home_directory + '/Library/Mobile Documents/com~apple~CloudDocs/Mitigation_project/SEAR_A'
                           '/Outputs/Plotting_outputs/heatmap_pv.png', dpi=300)

    plt.show()

def make_heatmap(f, tier, num, newcmp):

    heat_map_df = pd.read_csv(Plotting. plotting_output + 'heatmap_diff_out_tier_' + str(tier) + '.csv',
                              index_col='countries')
    heat_map_df = heat_map_df.drop('CT_N_RP0.0', axis=1)
    axarr = f.add_subplot(1, 3, num)
    #  plot seaborn heatmap
    # sns.heatmap(heat_map_df, cmap='YlGnBu', ax=axarr, cbar=False, annot=True, fmt='.0f', annot_kws={"size": 12})
    sns.heatmap(heat_map_df, cmap=newcmp, ax=axarr, cbar=False, linewidth=.05, annot=True, fmt='.2f',
                annot_kws={"size": 8}, vmin=-0.0, vmax=0.45)

    if tier > 2:
        axarr.yaxis.set_ticklabels([])
        axarr.set_ylabel('')
    axarr.set_title('Tier ' + str(tier))
    return f

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



"""
    input_list = [input_1, input_2, input_3]
    ax_list = [ax1, ax2, ax3]
    # Loop for each pie on subplot
    for pie in range(0, 3):
        x = []
        y = []
        for mode in modes:
            print(mode)
            mode_column = input_list[pie][mode + '_percentage_pop']
            if mode_column[0] == 0:
                pass
            else:
                y.append(mode_column[0] * 100)
                x.append(mode)
        ax_list[pie] = fig.add_axes([0.2, 0.4, 0.1, 0.1])  # [lowerCorner_x, lowerCorner_y, width, height]

         = plt.pie(y, colors=colors_pies)

"""