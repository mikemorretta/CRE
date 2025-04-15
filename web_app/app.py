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

def calculate_lease_payments(inputs):
    try:
        # Parse inputs
        year1_rent_psf = float(inputs['year1_rent_psf'])
        lease_term_months = int(inputs['lease_term_months'])
        annual_escalation_pct = float(inputs['annual_escalation_pct']) / 100
        rentable_sf = float(inputs['rentable_sf'])
        free_rent_months = int(inputs['free_rent_months'])
        ti_psf = float(inputs['ti_psf'])
        lc_psf = float(inputs['lc_psf'])
        discount_rate = float(inputs['discount_rate']) / 100

        # Calculate monthly payments
        monthly_payments = []
        for month in range(lease_term_months):
            year = month // MONTHS_PER_YEAR
            monthly_rate = year1_rent_psf * (1 + annual_escalation_pct) ** year
            if month < free_rent_months:
                monthly_payments.append(0)
            else:
                monthly_payments.append(monthly_rate * rentable_sf)

        # Calculate costs
        total_ti = ti_psf * rentable_sf
        total_lc = lc_psf * rentable_sf
        total_cost = total_ti + total_lc

        # Calculate NPV
        monthly_discount_rate = (1 + discount_rate) ** (1/MONTHS_PER_YEAR) - 1
        npv = sum(payment / ((1 + monthly_discount_rate) ** (i+1)) 
                 for i, payment in enumerate(monthly_payments))

        # Calculate payback period
        cumulative_cash_flow = 0
        payback_month = None
        for month, payment in enumerate(monthly_payments, 1):
            cumulative_cash_flow += payment
            if cumulative_cash_flow >= total_cost and payback_month is None:
                payback_month = month

        # Create annual summary
        annual_summary = {}
        for year in range(lease_term_months // MONTHS_PER_YEAR + 1):
            start_month = year * MONTHS_PER_YEAR
            end_month = min((year + 1) * MONTHS_PER_YEAR, lease_term_months)
            year_payments = monthly_payments[start_month:end_month]
            
            gross = sum(year_payments)
            abatement = sum(1 for p in year_payments if p == 0) * (year1_rent_psf * (1 + annual_escalation_pct) ** year * rentable_sf / MONTHS_PER_YEAR)
            net = gross - abatement
            
            annual_summary[str(year + 1)] = {
                'gross': gross,
                'abatement': abatement,
                'net': net,
                'gross_psf': gross / rentable_sf,
                'net_psf': net / rentable_sf,
                'net_per_month': net / len(year_payments) if year_payments else 0
            }

        # Create plot data
        plot_data = {
            'data': [{
                'x': list(range(1, lease_term_months + 1)),
                'y': [p / rentable_sf for p in monthly_payments],
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Monthly Rent PSF'
            }],
            'layout': {
                'title': 'Monthly Rent PSF Over Lease Term',
                'xaxis': {'title': 'Month'},
                'yaxis': {'title': 'Rent PSF'},
                'showlegend': True
            }
        }

        return {
            'success': True,
            'lease_summary': {
                'total_cost': total_cost,
                'total_cost_psf': total_cost / rentable_sf,
                'npv': npv,
                'payback_month': payback_month,
                'total_ti': total_ti,
                'total_ti_psf': ti_psf,
                'total_lc': total_lc,
                'total_lc_psf': lc_psf
            },
            'annual_summary': annual_summary,
            'plot': plot_data
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Error in calculate_lease_payments: {str(e)}"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        inputs = request.get_json()
        result = calculate_lease_payments(inputs)
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
        result = calculate_lease_payments(inputs)
        
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