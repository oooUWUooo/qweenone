-- Initialize database for Agentic Architecture System

-- Create table for agent states
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'idle',
    capabilities JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for tasks
CREATE TABLE IF NOT EXISTS tasks (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 1,
    status VARCHAR(50) DEFAULT 'pending',
    assigned_to VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create table for task execution history
CREATE TABLE IF NOT EXISTS execution_history (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255),
    agent_id VARCHAR(255),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration FLOAT,
    status VARCHAR(50),
    result JSONB,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_execution_task_id ON execution_history(task_id);
CREATE INDEX IF NOT EXISTS idx_execution_agent_id ON execution_history(agent_id);

-- Insert default agent types if they don't exist
INSERT INTO agents (id, name, type, description) 
SELECT 
    gen_random_uuid()::text,
    'Default Agent',
    'default',
    'Default agent for general tasks'
WHERE NOT EXISTS (SELECT 1 FROM agents WHERE type = 'default');