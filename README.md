# Pipeliner - Booking Target Revenue Probability App

This is a Streamlit application that helps estimate the probability of attaining a target revenue from a 'basket' of sales deals within a given time period.

## Project Structure

```
pipeliner
├── app
│   ├── __init__.py
│   ├── data_utils.py
│   ├── simulation.py
│   └── visualization.py
├── app.py
├── cli.py
├── one_file_app.py
├── pipeline.csv
├── README.md
├── requirements.txt
├── revenue_distribution.png
└── tests
    ├── __init__.py
    ├── test_data_utils.py
    └── test_simulation.py
```

## Installation

1.  Clone the repository: `git clone <repository-url>`
2.  Navigate to the project directory: `cd pipeliner`
3.  Install the dependencies: `pip install -r requirements.txt`

## Streamlit Usage

1.  Run the Streamlit app: `streamlit run app.py`
2.  Follow the instructions in the app to upload your deals CSV and run the simulations.

## CMDLINE Usage
1. python cli -f pipepline.csv -r 1000000

## Testing

1.  Navigate to the project directory: `cd pipeliner`
2.  Run the tests: `pytest tests/`

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## License

MIT
