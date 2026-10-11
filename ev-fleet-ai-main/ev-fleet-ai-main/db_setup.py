"""Initialize the ev_fleet MySQL schema and insert dummy test data."""

from __future__ import annotations

import os
from typing import Any

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error, MySQLConnection

load_dotenv()

VEHICLES: list[tuple[Any, ...]] = [
    (
        "Ather 450X",
        "3.7 kWh LFP A-Grade",
        146,
        1499.00,
        4999.00,
        "Available",
    ),
    (
        "Ola S1 Pro",
        "3.97 kWh NMC",
        181,
        1699.00,
        5499.00,
        "Rented",
    ),
    (
        "TVS iQube ST",
        "1.8-3.6 kWh LFP A-Grade",
        100,
        1299.00,
        4299.00,
        "Charging",
    ),
    (
        "Bajaj Chetak",
        "3.0 kWh LFP A-Grade",
        127,
        1399.00,
        4599.00,
        "Maintenance",
    ),
    (
        "Simple One",
        "4.8 kWh LFP A-Grade",
        212,
        1899.00,
        5999.00,
        "Available",
    ),
]

RIDERS: list[tuple[Any, ...]] = [
    ("Priya Sharma", "9876543210", 2, "Monthly", "Verified", "2024-11-02"),
    ("Arjun Mehta", "9123456780", 1, "Weekly", "Pending", "2025-01-15"),
    ("Kavya Iyer", "9988776655", 3, "Monthly", "Verified", "2024-08-21"),
    ("Rohan Das", "9012345678", None, "Weekly", "Rejected", "2025-03-08"),
    ("Neha Reddy", "9765432109", 5, "Monthly", "Verified", "2024-12-19"),
]


def _connect(database: str | None = None) -> MySQLConnection:
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=database,
    )


def setup_database() -> None:
    db_name = os.getenv("MYSQL_DATABASE", "ev_fleet")
    connection: MySQLConnection | None = None
    try:
        connection = _connect()
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cursor.execute(f"USE `{db_name}`")

        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        cursor.execute("DROP TABLE IF EXISTS riders")
        cursor.execute("DROP TABLE IF EXISTS vehicles")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

        cursor.execute(
            """
            CREATE TABLE vehicles (
                vehicle_id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(50) NOT NULL,
                battery_type VARCHAR(50) DEFAULT '1.8-3.6 kWh LFP A-Grade',
                range_per_charge_km INT,
                weekly_plan_rate DECIMAL(8, 2),
                monthly_plan_rate DECIMAL(8, 2),
                status ENUM('Available', 'Rented', 'Maintenance', 'Charging')
                    DEFAULT 'Available',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE riders (
                rider_id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                phone_number VARCHAR(15),
                assigned_vehicle_id INT,
                plan_type ENUM('Weekly', 'Monthly'),
                verification_status ENUM('Pending', 'Verified', 'Rejected')
                    DEFAULT 'Pending',
                joined_date DATE,
                FOREIGN KEY (assigned_vehicle_id) REFERENCES vehicles(vehicle_id)
            )
            """
        )

        cursor.executemany(
            """
            INSERT INTO vehicles (
                model_name, battery_type, range_per_charge_km,
                weekly_plan_rate, monthly_plan_rate, status
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            VEHICLES,
        )
        cursor.executemany(
            """
            INSERT INTO riders (
                full_name, phone_number, assigned_vehicle_id,
                plan_type, verification_status, joined_date
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            RIDERS,
        )
        connection.commit()
        print(f"Database `{db_name}` initialized with 5 vehicles and 5 riders.")
    except Error as exc:
        print(f"MySQL setup failed: {exc}")
        raise
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    setup_database()
