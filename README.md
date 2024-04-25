# Management of trading bots

## Getting started

## Migrations

Generate a migration based on changes

```shell
alembic revision --autogenerate -m "message"
```

Setup current revision without database changes

```shell
alembic stamp fe791c5083ef
```

Migrate to last version

```shell
alembic upgrade head
```

Revert

```shell
alembic downgrade -1
```

## Testing

## Deployment
