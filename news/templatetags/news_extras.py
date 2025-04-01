from django import template
from django.utils import timezone
from django.template.defaultfilters import stringfilter
from datetime import datetime
import json
from django.core.serializers.json import DjangoJSONEncoder

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    if not dictionary or not isinstance(dictionary, dict):
        return key
    return dictionary.get(key, key)

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def date(value, format_string):
    """Format a date using the provided format string."""
    if not value:
        return ''
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            try:
                value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    value = datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    return value
    return value.strftime(format_string) if isinstance(value, datetime) else value

@register.filter(name='json_encode')
def json_encode(value):
    """Convert a value to JSON string"""
    return json.dumps(value, cls=DjangoJSONEncoder)
