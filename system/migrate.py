#!/usr/bin/env python3
"""
Database migration helper script.
Run this to initialize the database schema.

Usage:
    python migrate.py init      # Generate Prisma client
    python migrate.py create    # Create migration
    python migrate.py deploy    # Apply migrations
    python migrate.py status    # Check migration status
    python migrate.py push      # Push schema (dev only)
    python migrate.py seed      # Seed database
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    return result


def init_prisma():
    """Initialize Prisma client."""
    print("=== Initializing Prisma Client ===")
    run_command("prisma generate")
    print("\nPrisma client generated successfully!")


def create_migration():
    """Create a new Prisma migration."""
    print("=== Creating Prisma Migration ===")
    run_command("prisma migrate dev --name init")
    print("\nMigration created successfully!")


def deploy_migration():
    """Deploy Prisma migrations to production."""
    print("=== Deploying Prisma Migrations ===")
    run_command("prisma migrate deploy")
    print("\nMigrations deployed successfully!")


def check_status():
    """Check Prisma migration status."""
    print("=== Checking Migration Status ===")
    run_command("prisma migrate status")


def push_schema():
    """Push schema to database (for development)."""
    print("=== Pushing Schema to Database ===")
    run_command("prisma db push")
    print("\nSchema pushed successfully!")


def seed_database():
    """Seed the database with initial data."""
    print("=== Seeding Database ===")
    seed_file = Path("prisma/seed.py")
    if not seed_file.exists():
        print("No seed file found at prisma/seed.py")
        print("Creating seed file...")
        seed_content = '''#!/usr/bin/env python3
"""Database seed script for Criminal Network Analysis System."""

import asyncio
from src.db import get_client, disconnect


async def seed():
    """Seed the database with initial data."""
    client = await get_client()
    
    try:
        # Create default police station
        station = await client.policestation.create(
            data={"name": "Central Police Station", "district": "Central Delhi"}
        )
        print(f"Created police station: {station.name}")
        
        # Create default admin user
        admin = await client.user.create(
            data={
                "email": "admin@delhi-police.gov.in",
                "name": "Admin",
                "passwordHash": "$2b$12$placeholder_hash",
                "role": "ADMIN",
                "status": "ACTIVE",
                "policeStationId": station.id,
            }
        )
        print(f"Created admin user: {admin.name}")
        
        print("Database seeded successfully!")
        
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(seed())
'''
        seed_file.write_text(seed_content)
        print(f"Created seed file: {seed_file}")
    run_command("python prisma/seed.py")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    commands = {
        "init": init_prisma,
        "create": create_migration,
        "deploy": deploy_migration,
        "status": check_status,
        "push": push_schema,
        "seed": seed_database,
    }
    
    if command not in commands:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
    
    commands[command]()


if __name__ == "__main__":
    main()