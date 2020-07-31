function test_map_function_generic(str_input_filename)

% test mapping function

% get some lines from giant file
%head -n 1000 loc-gowalla_totalCheckins.txt  > test.tab

% load this truncated data
fn_data = importdata(str_input_filename);
fn_data.data(1:10,1)
size(fn_data.data)


% plot on maps
% California map axes
figure; ax = usamap('california');
setm(ax, 'FFaceColor', [.5 .7 .9])
title('California map')

% read shapefile of US states with names and locations
states = geoshape(shaperead('usastatehi.shp', 'UseGeoCoords', true));

% display map
geoshow(states, 'Parent',ax)

% iterate through file and get lat, long and plot on map
for iCount = 1:size(fn_data.data,1)
	lat  = fn_data.data(iCount,1);
	lon  = fn_data.data(iCount,2);
	%lat = 37.773972
	%lon = -122.431297

	linem(lat, lon, 'LineStyle','none', 'LineWidth',2, 'Color','r', ...
    		'Marker','x', 'MarkerSize',10)
end

% save final map
saveas(gcf,'san_francisco_generic.eps', 'psc2')



