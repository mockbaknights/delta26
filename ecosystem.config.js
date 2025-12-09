// ecosystem.config.js
// PM2 configuration for running the Delta26 live engine
// Important: "script" must point to the .py file, not python3 itself.

module.exports = {
  apps: [
    {
      name: "delta26-engine",
      script: "live/live_engine.py",        // <-- RUN THIS PYTHON SCRIPT
      interpreter: "/usr/bin/python3",      // <-- USE THIS PYTHON INTERPRETER

      // Environment passed to the Python process
      env: {
        POLYGON_API_KEY: process.env.POLYGON_API_KEY,
      },

      watch: false,
      max_restarts: 5,
      restart_delay: 5000,
      time: true,
    },
  ],
};

