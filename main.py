from src.pipeline.trainer_pipeline import TrainPipeline
import webbrowser
import os

if __name__ == "__main__":

    print("Starting Training Pipeline...")

    pipeline = TrainPipeline()

    best_model, report, best_model_name = pipeline.start_training() # type: ignore

    print("Training Completed")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Training Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f4f4f4;
            }}
            h1 {{
                color: #333;
            }}
            .report {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .best-model {{
                background-color: #fff3cd;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="report">
            <h1>Model Training Results</h1>
            <h2>Best Model: <span style="color: #4CAF50;">{best_model_name}</span></h2>
            <h3>Model Performance Metrics</h3>
            <table>
                <tr>
                    <th>Model Name</th>
                    <th>R² Score</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>MAPE</th>
                </tr>
    """
    
    for model_name, metrics in report.items():
        r2 = metrics['r2']
        rmse = metrics['rmse']
        mae = metrics['mae']
        mape = metrics['mape']
        if model_name == best_model_name:
            html_content += f'<tr class="best-model"><td>{model_name}</td><td>{r2:.4f}</td><td>{rmse:.4f}</td><td>{mae:.4f}</td><td>{mape:.4f}</td></tr>'
        else:
            html_content += f'<tr><td>{model_name}</td><td>{r2:.4f}</td><td>{rmse:.4f}</td><td>{mae:.4f}</td><td>{mape:.4f}</td></tr>'
    
    html_content += """
            </table>
            <p><em>Training completed successfully!</em></p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file
    html_path = "artifacts/training_results.html"
    os.makedirs("artifacts", exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html_content)
    
    # Open in browser
    webbrowser.open("file://" + os.path.abspath(html_path))
    print(f"Results saved to: {html_path}")