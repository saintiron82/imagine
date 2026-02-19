/**
 * LoginPage — login / register with invite code.
 *
 * Dark theme matching the existing app design.
 */

import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { setServerUrl as setClientServerUrl } from '../api/client';
import { LogIn, UserPlus, Server, Eye, EyeOff, CheckCircle, XCircle } from 'lucide-react';

export default function LoginPage() {
  const { login, register, error, checkServerHealth } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [serverUrl, setServerUrlLocal] = useState(
    localStorage.getItem('imagine-server-url') || ''
  );
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [serverStatus, setServerStatus] = useState(null); // null | 'checking' | 'ok' | 'error'
  const [serverError, setServerError] = useState('');

  const handleCheckServer = async () => {
    if (!serverUrl.trim()) return;
    setServerStatus('checking');
    setServerError('');

    // Temporarily set server URL for health check
    setClientServerUrl(serverUrl.trim());

    const result = await checkServerHealth();
    if (result.ok) {
      setServerStatus('ok');
    } else {
      setServerStatus('error');
      setServerError(result.error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    const trimmedUrl = serverUrl.trim();
    let success;

    if (mode === 'login') {
      success = await login({ email, password, serverUrl: trimmedUrl });
    } else {
      success = await register({
        invite_code: inviteCode,
        username,
        email,
        password,
        serverUrl: trimmedUrl,
      });
    }

    setSubmitting(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="w-full max-w-md p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Imagine</h1>
          <p className="text-gray-400 text-sm">{t('auth.subtitle')}</p>
        </div>

        {/* Language toggle */}
        <div className="flex justify-center mb-6 gap-2">
          {availableLocales.map((loc) => (
            <button
              key={loc}
              onClick={() => setLocale(loc)}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                locale === loc
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {loc === 'en-US' ? 'EN' : 'KR'}
            </button>
          ))}
        </div>

        {/* Card */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          {/* Tab toggle */}
          <div className="flex mb-6 bg-gray-900 rounded-lg p-1">
            <button
              onClick={() => setMode('login')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                mode === 'login'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <LogIn size={16} />
              {t('auth.login')}
            </button>
            <button
              onClick={() => setMode('register')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                mode === 'register'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <UserPlus size={16} />
              {t('auth.register')}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Server URL */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.server_url')}</label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Server size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                  <input
                    type="url"
                    value={serverUrl}
                    onChange={(e) => {
                      setServerUrlLocal(e.target.value);
                      setServerStatus(null);
                    }}
                    placeholder="http://192.168.1.10:8000"
                    className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <button
                  type="button"
                  onClick={handleCheckServer}
                  disabled={!serverUrl.trim() || serverStatus === 'checking'}
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-xs text-gray-300 hover:bg-gray-600 disabled:opacity-50 transition-colors"
                >
                  {serverStatus === 'checking' ? '...' : t('auth.check')}
                </button>
              </div>
              {serverStatus === 'ok' && (
                <div className="flex items-center gap-1 mt-1 text-xs text-green-400">
                  <CheckCircle size={12} /> {t('auth.server_connected')}
                </div>
              )}
              {serverStatus === 'error' && (
                <div className="flex items-center gap-1 mt-1 text-xs text-red-400">
                  <XCircle size={12} /> {serverError}
                </div>
              )}
            </div>

            {/* Invite Code (register only) */}
            {mode === 'register' && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('auth.invite_code')}</label>
                <input
                  type="text"
                  value={inviteCode}
                  onChange={(e) => setInviteCode(e.target.value)}
                  placeholder={t('auth.invite_code_placeholder')}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>
            )}

            {/* Username (register only) */}
            {mode === 'register' && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('auth.username')}</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder={t('auth.username_placeholder')}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>
            )}

            {/* Email */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.email')}</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@example.com"
                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                required
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.password')}</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-3 py-2 pr-10 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  required
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={submitting || !serverUrl.trim()}
              className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              {submitting
                ? '...'
                : mode === 'login'
                  ? t('auth.login')
                  : t('auth.register')}
            </button>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-600 mt-4">
          Imagine Server v4.0
        </p>
      </div>
    </div>
  );
}
