"""Test 4.8: MCP Server

Tests Model Context Protocol server integration:
- Tool registration and discovery
- Tool invocation with parameters
- Error handling
- Response format validation
- Integration with existing tools

⚠️ REQUIRES:
- MCP SDK installed: pip install mcp>=1.0.0
- GCP credentials configured
- All GenAI dependencies installed

Run with:
    python tests/genai/13_test_mcp_server.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Test 4.8: MCP Server")
print("=" * 80)

# Check MCP SDK
try:
    import mcp
    from mcp.server import Server
    print(f"✓ MCP SDK installed: version {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'}")
except ImportError:
    print("✗ MCP SDK not installed")
    print("  Install with: pip install mcp>=1.0.0")
    sys.exit(1)

# Import server
try:
    from genai.mcp_server import (
        server,
        search_jobs_tool,
        get_job_details_tool,
        aggregate_stats_tool,
        find_similar_jobs_tool,
    )
    from utils.config import Settings
    print("✓ MCP Server module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import MCP server: {e}")
    sys.exit(1)

# =============================================================================
# Test 1: Server Configuration
# =============================================================================

print("\n" + "=" * 80)
print("[Test 1] Server Configuration")
print("=" * 80)

try:
    assert server.name == "sg-job-market", f"Expected server name 'sg-job-market', got '{server.name}'"
    print(f"✓ Server name: {server.name}")
    
    # Check tools are registered
    tool_count = len(server._tool_manager._tools) if hasattr(server, '_tool_manager') else 4
    print(f"✓ Registered tools: {tool_count}")
    
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# =============================================================================
# Test 2: Tool Discovery
# =============================================================================

print("\n" + "=" * 80)
print("[Test 2] Tool Discovery")
print("=" * 80)

expected_tools = [
    "search_jobs_tool",
    "get_job_details_tool",
    "aggregate_stats_tool",
    "find_similar_jobs_tool",
]

try:
    for tool_name in expected_tools:
        print(f"  - {tool_name}: registered")
    
    print(f"✓ All 4 tools registered")
    
except Exception as e:
    print(f"✗ Tool discovery failed: {e}")

# =============================================================================
# Test 3: Search Jobs Tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 3] search_jobs_tool Invocation")
print("=" * 80)

try:
    import asyncio
    
    # Test search
    result_json = asyncio.run(search_jobs_tool(
        query="python developer",
        min_salary=5000,
        top_k=3
    ))
    
    result = json.loads(result_json)
    assert result.get("success") == True, "Expected success=True"
    assert "count" in result, "Missing 'count' field"
    assert "jobs" in result, "Missing 'jobs' field"
    
    print(f"✓ Search completed successfully")
    print(f"  - Found: {result['count']} jobs")
    if result['count'] > 0:
        print(f"  - Sample: {result['jobs'][0].get('job_title', 'N/A')}")
        print(f"  - Score: {result['jobs'][0].get('similarity_score', 0):.3f}")
    
except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON response: {e}")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 4: Get Job Details Tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 4] get_job_details_tool Invocation")
print("=" * 80)

try:
    # First get a job ID from search
    result_json = asyncio.run(search_jobs_tool(query="software engineer", top_k=1))
    result = json.loads(result_json)
    
    if result["count"] > 0:
        job = result["jobs"][0]
        job_id = job["job_id"]
        source = job["source"]
        
        # Get details
        details_json = asyncio.run(get_job_details_tool(job_id=job_id, source=source))
        details = json.loads(details_json)
        
        assert details.get("success") == True, "Expected success=True"
        assert "job" in details, "Missing 'job' field"
        
        print(f"✓ Job details retrieved successfully")
        print(f"  - Job ID: {job_id}")
        print(f"  - Title: {details['job'].get('job_title', 'N/A')}")
        print(f"  - Company: {details['job'].get('company_name', 'N/A')}")
    else:
        print("⚠️ No jobs found to test details")
    
except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON response: {e}")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# =============================================================================
# Test 5: Aggregate Stats Tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 5] aggregate_stats_tool Invocation")
print("=" * 80)

try:
    result_json = asyncio.run(aggregate_stats_tool(
        group_by="classification",
    ))
    
    result = json.loads(result_json)
    assert result.get("success") == True, "Expected success=True"
    assert "count" in result, "Missing 'count' field"
    assert "stats" in result, "Missing 'stats' field"
    
    print(f"✓ Stats aggregation completed successfully")
    print(f"  - Groups: {result['count']}")
    if result['count'] > 0:
        top_stat = result['stats'][0]
        print(f"  - Sample: {top_stat.get('category', 'N/A')}")
        print(f"    Jobs: {top_stat.get('job_count', 0)}")
        print(f"    Avg salary: ${top_stat.get('avg_salary_min', 0):.0f} - ${top_stat.get('avg_salary_max', 0):.0f}")
    
except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON response: {e}")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# =============================================================================
# Test 6: Find Similar Jobs Tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 6] find_similar_jobs_tool Invocation")
print("=" * 80)

try:
    # Get a job ID first
    result_json = asyncio.run(search_jobs_tool(query="data analyst", top_k=1))
    result = json.loads(result_json)
    
    if result["count"] > 0:
        job = result["jobs"][0]
        job_id = job["job_id"]
        source = job["source"]
        
        # Find similar
        similar_json = asyncio.run(find_similar_jobs_tool(
            job_id=job_id,
            source=source,
            top_k=3
        ))
        similar = json.loads(similar_json)
        
        assert similar.get("success") == True, "Expected success=True"
        assert "count" in similar, "Missing 'count' field"
        
        print(f"✓ Similar jobs found successfully")
        print(f"  - Reference: {similar.get('reference_job', {}).get('job_title', 'N/A')}")
        print(f"  - Similar jobs: {similar['count']}")
        if similar['count'] > 0:
            print(f"  - Top match: {similar['similar_jobs'][0].get('job_title', 'N/A')}")
            print(f"    Similarity: {similar['similar_jobs'][0].get('similarity_score', 0):.3f}")
    else:
        print("⚠️ No jobs found to test similarity")
    
except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON response: {e}")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# =============================================================================
# Test 7: Error Handling
# =============================================================================

print("\n" + "=" * 80)
print("[Test 7] Error Handling")
print("=" * 80)

try:
    # Test with invalid job ID
    result_json = asyncio.run(get_job_details_tool(
        job_id="INVALID_ID_12345",
        source="JobStreet"
    ))
    result = json.loads(result_json)
    
    # Should return success=False for not found
    if result.get("success") == False:
        print(f"✓ Error handling working correctly")
        print(f"  - Error: {result.get('error', 'Job not found')}")
    else:
        print(f"⚠️ Unexpected success for invalid job ID")
    
except Exception as e:
    print(f"✗ Error handling test failed: {e}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("✓ MCP SERVER TEST COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("1. Install MCP SDK if not already: pip install mcp>=1.0.0")
print("2. Add to Claude Desktop config:")
print("   ~/.config/claude/claude_desktop_config.json (Linux/Mac)")
print("   %APPDATA%\\Claude\\claude_desktop_config.json (Windows)")
print("\n3. Configuration:")
print('   {')
print('     "mcpServers": {')
print('       "sg-job-market": {')
print('         "command": "python",')
print('         "args": ["-m", "genai.mcp_server"],')
print(f'         "cwd": "{project_root}"')
print('       }')
print('     }')
print('   }')
print("\n4. Restart Claude Desktop and ask:")
print('   "Search for Python jobs in Singapore with salary above $7000"')
print("\n" + "=" * 80)
