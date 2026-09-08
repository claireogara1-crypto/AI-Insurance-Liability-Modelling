# AI Liability Risk Forecasting

This project explores the forecasting of emerging AI liability risks in the insurance industry using incidents from the AI Incident Database (AIID).

## Data

The project uses data downloaded from the [AI Incident Database (AIID)]([https://incidentdatabase.ai/research/snapshots/])).

The exact screenshot used is from 2026-08-31 10:11 AM.

Download the AIID dataset and place the raw file in the `DataClassificationCode` folder.

## Preprocessing

To preprocess the data, run the 'incident_classifier' file and update the resulting incidents_classified.csv file in the 'Modelling' folder.

## Modelling

To run the models on the data, first run the '00_prepare_panel' file and then files 02 to 07 can be run in any order.

To compare the models, run file 08.
