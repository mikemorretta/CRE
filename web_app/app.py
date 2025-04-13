from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import io

app = Flask(__name__)

def calculate_lease_payments(inputs):
    # Extract inputs
    year1_rent_psf = inputs['year1_rent_psf']
    lease_term_months = inputs['lease_term_months']
    annual_escalation_pct = inputs['annual_escalation_pct']
    rentable_sf = inputs['rentable_sf']
    free_rent_months = inputs['free_rent_months']
    ti_psf = inputs['ti_psf']
    lc_psf = inputs['lc_psf']
    discount_rate = inputs['discount_rate']

    # Calculate monthly rents
    monthly_rents = []
    annual_rent_psf = []  # Store annualized rent PSF
    for month in range(lease_term_months):
        if month < free_rent_months:
            monthly_rents.append(0)
            annual_rent_psf.append(0)
        else:
            year = month // 12
            escalated_rent = year1_rent_psf * (1 + annual_escalation_pct/100) ** year
            monthly_rent = escalated_rent * rentable_sf / 12
            monthly_rents.append(monthly_rent)
            annual_rent_psf.append(escalated_rent)  # Store annualized rent PSF

    # Calculate cumulative payments
    cumulative_payments = np.cumsum(monthly_rents)

    # Calculate total cost (TI + LC)
    total_ti = ti_psf * rentable_sf
    total_lc = lc_psf * rentable_sf
    total_cost = total_ti + total_lc

    # Calculate NPV using custom function
    monthly_discount_rate = (1 + discount_rate/100) ** (1/12) - 1
    npv = sum(cf / ((1 + monthly_discount_rate) ** (i + 1)) for i, cf in enumerate(monthly_rents))

    # Calculate payback period
    payback_month = None
    for month, cum_payment in enumerate(cumulative_payments, 1):
        if cum_payment >= total_cost:
            payback_month = month
            break

    # Create annual summary
    annual_summary = {}
    for year in range(1, (lease_term_months // 12) + 1):
        start_month = (year - 1) * 12
        end_month = year * 12
        year_rents = monthly_rents[start_month:end_month]
        
        # Calculate gross rent for the year
        base_rent = year1_rent_psf * (1 + annual_escalation_pct/100) ** (year - 1)
        gross = base_rent * rentable_sf
        
        # Calculate free rent abatement
        free_months_in_year = max(0, min(12, free_rent_months - (year - 1) * 12))
        abatement = free_months_in_year * (base_rent * rentable_sf / 12)
        
        # Net rent is gross minus abatement, but never negative
        net = max(0, gross - abatement)
        
        annual_summary[year] = {
            'gross': gross,
            'abatement': abatement,
            'net': net,
            'gross_psf': gross / rentable_sf,
            'net_psf': net / rentable_sf,
            'net_per_month': net / 12
        }

    # Create payment schedule DataFrame
    dates = pd.date_range(start='2024-01-01', periods=lease_term_months, freq='ME')
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Create year labels for x-axis
    year_labels = []
    year_positions = []
    for i in range(lease_term_months):
        year = i // 12 + 1
        if i % 12 == 6:  # Center the label at month 6 of each year
            year_labels.append(f'Year {year}')
            year_positions.append(date_strings[i])
        else:
            year_labels.append('')
            year_positions.append('')

    payment_schedule = pd.DataFrame({
        'Date': date_strings,
        'Year Label': year_labels,
        'Monthly Rent': monthly_rents,
        'Cumulative Rent': cumulative_payments
    })

    # Create the plot data
    plot_data = [
        {
            'x': date_strings,
            'y': annual_rent_psf,
            'mode': 'lines',
            'name': 'Annual Rent PSF',
            'line': {'color': '#6c757d', 'width': 2, 'shape': 'hv'},  # Neutral gray
            'hovertemplate': 'Year %{customdata}<br>Annual Rent PSF: $%{y:.2f}',
            'customdata': [f'Year {i//12 + 1}' for i in range(len(date_strings))]
        },
        {
            'x': date_strings,
            'y': cumulative_payments.tolist(),
            'mode': 'lines',
            'name': 'Cumulative Rent',
            'line': {'color': '#495057', 'width': 2, 'dash': 'dash'},  # Darker gray
            'yaxis': 'y2',
            'hovertemplate': 'Year %{customdata}<br>Cumulative Rent: $%{y:,.0f}',
            'customdata': [f'Year {i//12 + 1}' for i in range(len(date_strings))]
        },
        {
            'x': [date_strings[0], date_strings[-1]],
            'y': [total_cost, total_cost],
            'mode': 'lines',
            'name': 'TI + LC Cost',
            'line': {'color': '#dc3545', 'width': 2, 'dash': 'dot'},  # Bootstrap red
            'yaxis': 'y2',
            'hovertemplate': 'TI + LC Cost: $%{y:,.0f}'
        }
    ]

    # Add shapes and annotations
    shapes = []
    annotations = []

    # Add free rent period highlight
    if free_rent_months > 0:
        shapes.append({
            'type': 'rect',
            'x0': date_strings[0],
            'x1': date_strings[free_rent_months],
            'y0': 0,
            'y1': 1,
            'yref': 'paper',
            'fillcolor': 'rgba(142, 142, 147, 0.4)',
            'opacity': 0.5,
            'layer': 'below',
            'line': {'width': 0}
        })
        annotations.append({
            'x': date_strings[free_rent_months // 2],
            'y': 0.5,
            'text': 'Free Rent Period',
            'showarrow': False,
            'font': {'size': 12, 'color': 'rgba(142, 142, 147, 0.9)'},
            'xref': 'x',
            'yref': 'paper'
        })

    # Add yearly gridlines
    for year in range(1, lease_term_months // 12 + 1):
        month_marker = year * 12
        if month_marker < len(date_strings):
            shapes.append({
                'type': 'line',
                'x0': date_strings[month_marker],
                'x1': date_strings[month_marker],
                'y0': 0,
                'y1': 1,
                'yref': 'paper',
                'line': {'color': 'rgba(0,0,0,0.1)', 'width': 1},
                'layer': 'below'
            })

    # Create layout
    plot_layout = {
        'template': 'none',
        'title': {
            'text': '',
            'font': {'size': 18, 'color': '#1d1d1f'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'title': {
                'text': 'Lease Year',
                'font': {'size': 14, 'color': '#1d1d1f'},
                'standoff': 20
            },
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'zerolinecolor': 'rgba(0,0,0,0.2)',
            'showline': True,
            'linecolor': 'rgba(0,0,0,0.2)',
            'mirror': True,
            'tickfont': {'color': '#1d1d1f'},
            'ticktext': year_labels,
            'tickvals': year_positions,
            'tickangle': 0,
            'gridwidth': 1,
            'griddash': 'dot',
            'range': [date_strings[0], date_strings[-1]],
            'domain': [0, 1],
            'dtick': 12  # Set major tick interval to 12 months
        },
        'yaxis': {
            'title': {
                'text': 'Annual Rent PSF ($)',
                'font': {'size': 14, 'color': '#1d1d1f'},
                'standoff': 20
            },
            'side': 'left',
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'zerolinecolor': 'rgba(0,0,0,0.2)',
            'showline': True,
            'linecolor': 'rgba(0,0,0,0.2)',
            'mirror': True,
            'tickfont': {'color': '#1d1d1f'},
            'range': [0, max(annual_rent_psf) * 1.1]
        },
        'yaxis2': {
            'title': {
                'text': 'Cumulative Rent ($)',
                'font': {'size': 14, 'color': '#1d1d1f'},
                'standoff': 20
            },
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False,
            'showline': True,
            'linecolor': 'rgba(0,0,0,0.2)',
            'mirror': True,
            'tickfont': {'color': '#1d1d1f'},
            'range': [0, max(cumulative_payments) * 1.1]
        },
        'legend': {
            'x': 0.01,
            'y': 0.99,
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': 'rgba(0,0,0,0.2)',
            'font': {'color': '#1d1d1f'}
        },
        'margin': {'l': 80, 'r': 80, 't': 40, 'b': 80},
        'height': 600,
        'width': None,
        'paper_bgcolor': 'rgba(255,255,255,0)',
        'plot_bgcolor': 'rgba(255,255,255,0)',
        'font': {
            'family': 'SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif',
            'color': '#1d1d1f'
        },
        'shapes': shapes,
        'annotations': annotations
    }

    return {
        'monthly_rents': monthly_rents,
        'cumulative_payments': cumulative_payments.tolist(),
        'total_cost': total_cost,
        'total_cost_psf': total_cost / rentable_sf,
        'total_ti': total_ti,
        'total_ti_psf': ti_psf,
        'total_lc': total_lc,
        'total_lc_psf': lc_psf,
        'npv': npv,
        'payback_month': payback_month,
        'annual_summary': annual_summary,
        'payment_schedule': payment_schedule.to_dict('records'),
        'plot': {
            'data': plot_data,
            'layout': plot_layout
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        inputs = request.get_json()
        if not inputs:
            return jsonify({
                'success': False,
                'error': 'No input data received'
            })
        
        # Validate required fields
        required_fields = ['year1_rent_psf', 'lease_term_months', 'annual_escalation_pct', 
                         'rentable_sf', 'free_rent_months', 'ti_psf', 'lc_psf', 'discount_rate']
        missing_fields = [field for field in required_fields if field not in inputs]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            })
        
        results = calculate_lease_payments(inputs)
        
        return jsonify({
            'success': True,
            'plot': results['plot'],
            'payment_schedule': results['payment_schedule'],
            'annual_summary': results['annual_summary'],
            'lease_summary': {
                'total_cost': results['total_cost'],
                'total_cost_psf': results['total_cost_psf'],
                'total_ti': results['total_ti'],
                'total_ti_psf': results['total_ti_psf'],
                'total_lc': results['total_lc'],
                'total_lc_psf': results['total_lc_psf'],
                'npv': results['npv'],
                'payback_month': results['payback_month']
            }
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")  # This will print to the server console
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        })

@app.route('/export_csv', methods=['POST'])
def export_csv():
    try:
        data = request.json
        results = calculate_lease_payments(data)
        
        # Create CSV data
        output = io.StringIO()
        results['payment_schedule'].to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='lease_schedule.csv'
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080) 