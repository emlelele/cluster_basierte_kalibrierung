# Database Operations with GeoPandas

This project includes Python scripts for managing database operations, establishing SSH
tunnels, and handling spatial data with GeoPandas. The scripts are designed to interact
with a PostgreSQL database, using SQLAlchemy for ORM capabilities, and set up SSH
tunnels for secure database access.

## Project Structure

- `mwe_db_access/`: Module containing the database access and SSH tunnel setup
functionalities.
  - `config/`: Configuration settings for database and SSH.
  - `db.py`: Functions related to database engine creation, schema registration, and
  session management.
  - `ssh.py`: Context manager for managing SSH tunnels.
- `main.py`: Main script that utilizes modules for performing database queries and
converting results to GeoDataFrame.

## Setup

```bash
mamba env create -f environment.yml
```
