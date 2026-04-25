// hooks/usePolling.js
import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Polls an async fetch function at a given interval.
 * Returns { data, loading, error, refresh }.
 */
export function usePolling(fetchFn, intervalMs = 5000, deps = []) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const result = await fetchFn();
      if (mountedRef.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) setError(err.message);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mountedRef.current = true;
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [refresh, intervalMs]);

  return { data, loading, error, refresh };
}

/**
 * One-shot fetch with loading/error state (no polling).
 */
export function useFetch(fetchFn, deps = []) {
  return usePolling(fetchFn, 0, deps);
}
