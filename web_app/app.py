from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import plotly
import json
import io
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Constants
MONTHS_PER_YEAR = 12

def calculate_lease_payments(year1_rent_psf, lease_term_months, annual_escalation_pct, 
                           rentable_sf, free_rent_months, ti_psf, lc_psf, discount_rate):
    # Parse inputs
    year1_rent_psf = float(year1_rent_psf)
    lease_term_months = int(lease_term_months)
    annual_escalation_pct = float(annual_escalation_pct) / 100
    rentable_sf = float(rentable_sf)
    free_rent_months = int(free_rent_months)
    ti_psf = float(ti_psf)
    lc_psf = float(lc_psf)
    discount_rate = float(discount_rate) / 100

    # Calculate monthly payments
    psf_rents = []
    monthly_rents = []
    cumulative_rent = 0
    cumulative_rents = []
    
    for month in range(1, lease_term_months + 1):
        # Calculate rent PSF for this month (annual rate)
        if month <= free_rent_months:
            psf_rent = 0
        else:
            # Calculate annual escalation
            years = (month - 1) // 12
            psf_rent = year1_rent_psf * (1 + annual_escalation_pct) ** years
        
        # Calculate monthly rent (divide annual PSF by 12 for monthly amount)
        monthly_rent = (psf_rent / 12) * rentable_sf
        
        # Update cumulative rent
        cumulative_rent += monthly_rent
        
        psf_rents.append(psf_rent)
        monthly_rents.append(monthly_rent)
        cumulative_rents.append(cumulative_rent)

    # Calculate total costs
    total_ti = ti_psf * rentable_sf
    total_lc = lc_psf * rentable_sf
    total_cost = total_ti + total_lc

    # Calculate NPV
    monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1
    npv = -total_cost
    for month, rent in enumerate(monthly_rents, 1):
        npv += rent / ((1 + monthly_discount_rate) ** month)

    # Calculate payback period
    payback_month = None
    cumulative_cash_flow = -total_cost
    for month, rent in enumerate(monthly_rents, 1):
        cumulative_cash_flow += rent
        if cumulative_cash_flow >= 0 and payback_month is None:
            payback_month = month

    # Create plot data
    months = list(range(1, lease_term_months + 1))
    plot_data = {
        'data': [
            {
                'x': months,
                'y': psf_rents,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Annual Rent PSF',
                'yaxis': 'y1',
                'line': {
                    'shape': 'hv',
                    'width': 2,
                    'color': '#1a1a1a'
                }
            },
            {
                'x': months,
                'y': cumulative_rents,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Cumulative Rent',
                'yaxis': 'y2',
                'line': {
                    'width': 2,
                    'color': '#003366'
                }
            }
        ],
        'layout': {
            'title': {
                'text': 'Lease Payment Schedule',
                'font': {
                    'size': 20,
                    'color': '#1a1a1a',
                    'family': 'Arial, sans-serif'
                },
                'x': 0.5,
                'y': 0.95
            },
            'xaxis': {
                'title': {
                    'text': 'Month',
                    'font': {
                        'size': 14,
                        'color': '#1a1a1a'
                    }
                },
                'showgrid': True,
                'gridcolor': '#e6e6e6',
                'zeroline': True,
                'zerolinecolor': '#e6e6e6',
                'range': [0, lease_term_months + 1],
                'dtick': 12,
                'tickformat': 'd',
                'tickfont': {
                    'size': 12,
                    'color': '#666666'
                },
                'showline': True,
                'linecolor': '#cccccc',
                'mirror': True,
                'automargin': True
            },
            'yaxis': {
                'title': {
                    'text': 'Annual Rent PSF',
                    'font': {
                        'size': 14,
                        'color': '#1a1a1a'
                    }
                },
                'showgrid': True,
                'gridcolor': '#e6e6e6',
                'zeroline': True,
                'zerolinecolor': '#e6e6e6',
                'range': [0, max(psf_rents) * 1.2],
                'tickprefix': '$',
                'tickformat': '.2f',
                'tickfont': {
                    'size': 12,
                    'color': '#666666'
                },
                'showline': True,
                'linecolor': '#cccccc',
                'mirror': True,
                'automargin': True
            },
            'yaxis2': {
                'title': {
                    'text': 'Cumulative Rent',
                    'font': {
                        'size': 14,
                        'color': '#1a1a1a'
                    }
                },
                'showgrid': False,
                'zeroline': True,
                'zerolinecolor': '#e6e6e6',
                'range': [0, max(cumulative_rents) * 1.2],
                'tickprefix': '$',
                'tickformat': ',.0f',
                'overlaying': 'y',
                'side': 'right',
                'tickfont': {
                    'size': 12,
                    'color': '#666666'
                },
                'showline': True,
                'linecolor': '#cccccc',
                'mirror': True,
                'automargin': True
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {
                'color': '#1a1a1a',
                'family': 'Arial, sans-serif'
            },
            'margin': {
                'l': 100,
                'r': 100,
                't': 100,  # Significantly increased bottom margin
                'b': 200,
                'pad': 8,
                'autoexpand': True
            },
            'height': 600,
            'showlegend': True,
            'legend': {
                'x': 1,
                'y': 1,
                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                'bordercolor': '#cccccc',
                'borderwidth': 1,
                'font': {
                    'size': 12,
                    'color': '#1a1a1a'
                }
            },
            'shapes': [
                {
                    'type': 'line',
                    'x0': 0,
                    'x1': lease_term_months,
                    'y0': total_cost,
                    'y1': total_cost,
                    'line': {
                        'color': '#990000',
                        'width': 2,
                        'dash': 'dot'
                    },
                    'yref': 'y2'
                }
            ],
            'annotations': [
                {
                    'x': lease_term_months,
                    'y': total_cost,
                    'text': f'TI/LC Cost: ${total_cost:,.0f} (${total_cost/rentable_sf:.2f}/SF)',
                    'showarrow': False,
                    'font': {
                        'size': 12,
                        'color': '#990000'
                    },
                    'bgcolor': 'rgba(255, 255, 255, 0.9)',
                    'bordercolor': '#cccccc',
                    'borderwidth': 1,
                    'borderpad': 4,
                    'xanchor': 'right',
                    'yanchor': 'bottom',
                    'yref': 'y2'
                }
            ]
        }
    }

    # Add free rent period shading
    if free_rent_months > 0:
        plot_data['layout']['shapes'].append({
            'type': 'rect',
            'x0': 0,
            'x1': free_rent_months,
            'y0': 0,
            'y1': max(psf_rents) * 1.2,
            'fillcolor': 'rgba(230, 230, 230, 0.3)',
            'line': {
                'width': 0
            },
            'layer': 'below'
        })

        # Add free rent annotation
        plot_data['layout']['annotations'].append({
            'x': free_rent_months / 2,
            'y': max(psf_rents) * 0.5,
            'text': f'Free Rent ({free_rent_months} months)',
            'showarrow': False,
            'font': {
                'size': 12,
                'color': '#666666'
            },
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#cccccc',
            'borderwidth': 1,
            'borderpad': 4,
            'layer': 'above'
        })

    # Add breakeven note if it exists
    if payback_month:
        # Add the note text
        plot_data['layout']['annotations'].append({
            'x': lease_term_months / 2,  # Center of the x-axis
            'y': 0,  # Bottom of the chart
            'text': f'Note: Cumulative rent received equals TI/LC cost in month {payback_month}',
            'showarrow': False,
            'font': {
                'size': 14,
                'color': '#1a1a1a',
                'family': 'Arial, sans-serif'
            },
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#cccccc',
            'borderwidth': 1,
            'borderpad': 8,
            'yanchor': 'top'
        })

    # Create annual summary
    annual_summary = {}
    for year in range(1, (lease_term_months // 12) + 1):
        start_month = (year - 1) * 12
        end_month = min(year * 12, lease_term_months)
        year_rents = monthly_rents[start_month:end_month]
        year_abatement = sum(1 for m in range(start_month, end_month) if m < free_rent_months)
        
        gross = sum(year_rents)
        abatement = sum(monthly_rents[start_month:start_month + year_abatement]) if year_abatement > 0 else 0
        net = gross - abatement
        
        annual_summary[year] = {
            'gross': gross,
            'abatement': abatement,
            'net': net,
            'gross_psf': gross / rentable_sf,
            'net_psf': net / rentable_sf,
            'net_per_month': net / 12
        }

    return {
        'success': True,
        'lease_summary': {
            'total_cost': total_cost,
            'total_cost_psf': total_cost / rentable_sf,
            'npv': npv,
            'payback_month': payback_month,
            'total_ti': total_ti,
            'total_ti_psf': total_ti / rentable_sf,
            'total_lc': total_lc,
            'total_lc_psf': total_lc / rentable_sf
        },
        'plot': plot_data,
        'annual_summary': annual_summary
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        inputs = request.get_json()
        result = calculate_lease_payments(inputs['year1_rent_psf'], inputs['lease_term_months'], inputs['annual_escalation_pct'], 
                                         inputs['rentable_sf'], inputs['free_rent_months'], inputs['ti_psf'], inputs['lc_psf'], inputs['discount_rate'])
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'details': f"Error in calculate endpoint: {str(e)}"
        })

@app.route('/export_csv', methods=['POST'])
def export_csv():
    try:
        inputs = request.get_json()
        result = calculate_lease_payments(inputs['year1_rent_psf'], inputs['lease_term_months'], inputs['annual_escalation_pct'], 
                                         inputs['rentable_sf'], inputs['free_rent_months'], inputs['ti_psf'], inputs['lc_psf'], inputs['discount_rate'])
        
        if not result['success']:
            return jsonify(result)

        # Create DataFrame for monthly payments
        months = list(range(1, int(inputs['lease_term_months']) + 1))
        monthly_payments = [p / float(inputs['rentable_sf']) for p in 
                          [0 if m <= int(inputs['free_rent_months']) else 
                           float(inputs['year1_rent_psf']) * 
                           (1 + float(inputs['annual_escalation_pct'])/100) ** ((m-1) // 12) 
                           for m in months]]
        
        df = pd.DataFrame({
            'Month': months,
            'Rent_PSF': monthly_payments
        })

        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'lease_payment_schedule_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'details': f"Error in export_csv endpoint: {str(e)}"
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 