import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from scipy.interpolate import splprep, splev
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# ======== INPUT DATA ========
points = np.array([[1835., 1193.],
       [1840., 1190.],
       [1845., 1190.],
       [1850., 1188.],
       [1855., 1186.],
       [1860., 1185.],
       [1865., 1184.],
       [1870., 1182.],
       [1875., 1181.],
       [1880., 1180.],
       [1885., 1179.],
       [1890., 1177.],
       [1895., 1176.],
       [1900., 1175.],
       [1905., 1174.],
       [1910., 1172.],
       [1915., 1171.],
       [1920., 1170.],
       [1925., 1168.],
       [1930., 1167.],
       [1935., 1166.],
       [1940., 1165.],
       [1945., 1164.],
       [1950., 1162.],
       [1955., 1161.],
       [1960., 1160.],
       [1965., 1158.],
       [1970., 1158.],
       [1975., 1156.],
       [1980., 1155.],
       [1985., 1153.],
       [1990., 1152.],
       [1995., 1151.],
       [2000., 1150.],
       [2005., 1149.],
       [2010., 1147.],
       [2015., 1147.],
       [2020., 1145.],
       [2025., 1144.],
       [2030., 1143.],
       [2035., 1142.],
       [2040., 1141.],
       [2045., 1139.],
       [2050., 1138.],
       [2055., 1137.],
       [2060., 1136.],
       [2065., 1136.],
       [2070., 1137.],
       [2075., 1137.],
       [2080., 1139.],
       [2085., 1140.],
       [2090., 1142.],
       [2094., 1146.],
       [2098., 1151.],
       [2100., 1156.],
       [2101., 1161.],
       [2102., 1166.],
       [2102., 1171.],
       [2103., 1176.],
       [2103., 1181.],
       [2103., 1186.],
       [2104., 1191.],
       [2104., 1196.],
       [2105., 1201.],
       [2105., 1206.],
       [2106., 1211.],
       [2107., 1216.],
       [2108., 1221.],
       [2108., 1226.],
       [2109., 1231.],
       [2109., 1236.],
       [2111., 1241.],
       [2112., 1246.],
       [2113., 1251.],
       [2114., 1256.],
       [2115., 1261.],
       [2117., 1266.],
       [2120., 1271.],
       [2125., 1276.],
       [2127., 1281.],
       [2132., 1286.],
       [2137., 1290.],
       [2142., 1294.],
       [2147., 1297.],
       [2152., 1300.],
       [2157., 1303.],
       [2162., 1305.],
       [2167., 1306.],
       [2172., 1307.],
       [2177., 1308.],
       [2182., 1308.],
       [2187., 1309.],
       [2192., 1309.],
       [2197., 1309.],
       [2202., 1308.],
       [2207., 1308.],
       [2212., 1308.],
       [2217., 1307.],
       [2222., 1306.],
       [2227., 1304.],
       [2232., 1304.],
       [2237., 1303.],
       [2242., 1302.],
       [2247., 1301.],
       [2252., 1300.],
       [2257., 1299.],
       [2262., 1298.],
       [2267., 1297.],
       [2272., 1296.],
       [2277., 1295.],
       [2282., 1293.],
       [2287., 1292.],
       [2292., 1291.],
       [2297., 1289.],
       [2302., 1288.],
       [2307., 1287.],
       [2312., 1285.],
       [2317., 1284.],
       [2322., 1282.],
       [2327., 1280.],
       [2332., 1278.],
       [2337., 1276.],
       [2342., 1272.],
       [2347., 1269.],
       [2352., 1265.],
       [2357., 1261.],
       [2362., 1257.],
       [2366., 1252.],
       [2369., 1248.],
       [2373., 1243.],
       [2376., 1238.],
       [2379., 1233.],
       [2381., 1228.],
       [2383., 1223.],
       [2384., 1218.],
       [2385., 1213.],
       [2385., 1208.],
       [2387., 1204.],
       [2387., 1199.],
       [2384., 1194.],
       [2383., 1189.],
       [2382., 1184.],
       [2381., 1179.],
       [2380., 1174.],
       [2378., 1169.],
       [2377., 1164.],
       [2376., 1159.],
       [2375., 1154.],
       [2374., 1149.],
       [2372., 1144.],
       [2371., 1139.],
       [2368., 1134.],
       [2367., 1129.],
       [2364., 1124.],
       [2361., 1119.],
       [2360., 1114.],
       [2360., 1109.],
       [2358., 1104.],
       [2356., 1099.],
       [2356., 1094.],
       [2355., 1089.],
       [2353., 1084.],
       [2353., 1079.],
       [2353., 1074.],
       [2353., 1069.],
       [2355., 1064.],
       [2360., 1059.],
       [2365., 1057.],
       [2370., 1055.],
       [2375., 1053.],
       [2380., 1052.],
       [2385., 1051.],
       [2390., 1050.],
       [2395., 1048.],
       [2400., 1047.],
       [2405., 1046.],
       [2410., 1044.],
       [2415., 1044.],
       [2420., 1042.],
       [2425., 1041.],
       [2430., 1040.],
       [2435., 1039.],
       [2440., 1038.],
       [2445., 1036.],
       [2450., 1035.],
       [2455., 1034.],
       [2460., 1033.],
       [2465., 1031.],
       [2470., 1030.],
       [2475., 1029.],
       [2480., 1028.],
       [2485., 1027.],
       [2490., 1026.],
       [2495., 1024.],
       [2500., 1023.],
       [2505., 1022.],
       [2510., 1021.],
       [2515., 1019.],
       [2520., 1018.],
       [2525., 1017.],
       [2530., 1016.],
       [2535., 1015.],
       [2540., 1014.],
       [2545., 1012.],
       [2550., 1011.],
       [2555., 1010.],
       [2560., 1009.],
       [2565., 1008.],
       [2570., 1006.],
       [2575., 1006.],
       [2580., 1004.],
       [2585., 1003.],
       [2590., 1002.],
       [2595., 1001.],
       [2600., 1000.],
       [2605.,  999.],
       [2610.,  999.],
       [2615.,  997.],
       [2620.,  994.],
       [2625.,  991.],
       [2628.,  988.]], dtype=float)

x, y = points[:, 0], points[:, 1]

# ======== Dash App ========
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Puzzle Piece Edge Analysis Tool"),

    html.Div([
        html.Label("Polynomial Order:"),
        dcc.Slider(1, 10, 1, value=2, id="poly-order", marks={i: str(i) for i in range(1, 11)})
    ], style={"width": "40%", "margin": "20px"}),

    html.Div([
        dcc.Checklist(
            id="show-options",
            options=[
                {"label": "Show Original Points", "value": "points"},
                {"label": "Show Spline Fit", "value": "spline"},
                {"label": "Show Linear Fit", "value": "linear"},
                {"label": "Show Polynomial Fit", "value": "poly"},
                {"label": "Show Max Deviations", "value": "dev"}
            ],
            value=["points", "spline", "linear", "poly", "dev"],
            inline=True
        )
    ], style={"margin": "10px"}),

    dcc.Graph(id="point-graph"),

    html.Div(id="stats-output", style={
        "whiteSpace": "pre-line",
        "margin": "20px",
        "fontFamily": "monospace",
        "backgroundColor": "#f8f8f8",
        "padding": "10px",
        "borderRadius": "8px"
    })
])

@app.callback(
    Output("point-graph", "figure"),
    Output("stats-output", "children"),
    Input("poly-order", "value"),
    Input("show-options", "value")
)
def update_graph(poly_order, show_opts):
    fig = go.Figure()

    # Base stats
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    total_length = np.sum(distances)
    avg_slope = (y[-1] - y[0]) / (x[-1] - x[0])
    linreg = linregress(x, y)

    # Fits
    lin_fit_y = linreg.slope * x + linreg.intercept
    poly_coeffs = np.polyfit(x, y, poly_order)
    poly_fit_y = np.polyval(poly_coeffs, x)

    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, len(x))
    spline_x, spline_y = splev(unew, tck)

    # Deviations
    dev_linear = y - lin_fit_y
    dev_poly = y - poly_fit_y
    dev_spline = y - spline_y

    max_dev_linear_idx = np.argmax(np.abs(dev_linear))
    max_dev_poly_idx = np.argmax(np.abs(dev_poly))
    max_dev_spline_idx = np.argmax(np.abs(dev_spline))

    stats_text = (
        f"Number of points: {len(points)}\n"
        f"Total path length: {total_length:.2f}\n"
        f"Average slope: {avg_slope:.4f}\n"
        f"Linear fit: y = {linreg.slope:.4f}x + {linreg.intercept:.2f} "
        f"(R²={linreg.rvalue**2:.4f})\n\n"
        f"Max deviation from linear fit: {dev_linear[max_dev_linear_idx]:.4f} "
        f"at {points[max_dev_linear_idx]}\n"
        f"Max deviation from polynomial fit (order {poly_order}): {dev_poly[max_dev_poly_idx]:.4f} "
        f"at {points[max_dev_poly_idx]}\n"
        f"Max deviation from spline fit: {dev_spline[max_dev_spline_idx]:.4f} "
        f"at {points[max_dev_spline_idx]}"
    )

    # Plot Original Points
    if "points" in show_opts:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers+lines",
            name="Original Points",
            marker=dict(size=6)
        ))

    # Spline Fit
    if "spline" in show_opts:
        unew_fine = np.linspace(0, 1, 500)
        x_smooth, y_smooth = splev(unew_fine, tck)
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth,
            mode="lines",
            name="Spline Fit"
        ))

    # Linear Fit
    if "linear" in show_opts:
        fig.add_trace(go.Scatter(
            x=x,
            y=lin_fit_y,
            mode="lines",
            name="Linear Fit",
            line=dict(dash="dash")
        ))

    # Polynomial Fit
    if "poly" in show_opts:
        fig.add_trace(go.Scatter(
            x=x,
            y=poly_fit_y,
            mode="lines",
            name=f"Poly Fit (order {poly_order})",
            line=dict(dash="dot")
        ))

    # Max Deviations
    if "dev" in show_opts:
        fig.add_trace(go.Scatter(
            x=[x[max_dev_linear_idx]], y=[y[max_dev_linear_idx]],
            mode="markers", name="Max Dev Linear",
            marker=dict(color="red", size=10, symbol="circle")
        ))
        fig.add_trace(go.Scatter(
            x=[x[max_dev_poly_idx]], y=[y[max_dev_poly_idx]],
            mode="markers", name="Max Dev Poly",
            marker=dict(color="blue", size=10, symbol="diamond")
        ))
        fig.add_trace(go.Scatter(
            x=[x[max_dev_spline_idx]], y=[y[max_dev_spline_idx]],
            mode="markers", name="Max Dev Spline",
            marker=dict(color="green", size=10, symbol="star")
        ))

    fig.update_layout(
        title="Puzzle Piece Edge Analysis",
        xaxis_title="X",
        yaxis_title="Y",
        hovermode="closest"
    )

    return fig, stats_text

if __name__ == "__main__":
    app.run(debug=True)
