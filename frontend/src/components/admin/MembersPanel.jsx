/**
 * MembersPanel — user/member management for admin.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import {
  listMembers, updateMemberRole, removeMember, deactivateMember, activateMember,
} from '../../api/admin';
import {
  Shield, ShieldOff, Trash2, CheckCircle, XCircle,
} from 'lucide-react';


export default function MembersPanel({ currentUser }) {
  const { t } = useLocale();
  const [members, setMembers] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listMembers();
      setMembers(data.members || []);
    } catch (e) {
      console.error('Failed to load members:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const isSelf = (m) => {
    if (!currentUser) return false;
    return m.id === currentUser.uid || m.id === currentUser.id || m.email === currentUser.email;
  };

  const handleRoleChange = async (m, newRole) => {
    if (isSelf(m)) return;
    try {
      await updateMemberRole(m.id, newRole);
      load();
    } catch (e) {
      console.error('Failed to update member role:', e);
    }
  };

  const handleToggleActive = async (m) => {
    if (isSelf(m)) return;
    try {
      if (m.is_active) {
        await deactivateMember(m.id);
      } else {
        await activateMember(m.id);
      }
      load();
    } catch (e) {
      console.error('Failed to toggle member status:', e);
    }
  };

  const handleRemove = async (m) => {
    if (isSelf(m)) return;
    if (!confirm(t('members.confirm_remove'))) return;
    try {
      await removeMember(m.id);
      load();
    } catch (e) {
      console.error('Failed to remove member:', e);
    }
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('members.title')}</h2>
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">{t('members.col_email')}</th>
              <th className="text-left px-4 py-3">{t('members.col_display_name')}</th>
              <th className="text-left px-4 py-3">{t('members.col_role')}</th>
              <th className="text-left px-4 py-3">{t('members.col_status')}</th>
              <th className="text-left px-4 py-3">{t('members.col_joined')}</th>
              <th className="text-right px-4 py-3">{t('members.col_actions')}</th>
            </tr>
          </thead>
          <tbody>
            {members.map((m) => (
              <tr key={m.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                <td className="px-4 py-3 font-medium">
                  {m.email}
                  {isSelf(m) && (
                    <span className="ml-2 text-xs text-blue-400">({t('members.cannot_change_self')})</span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-400">{m.display_name || '-'}</td>
                <td className="px-4 py-3">
                  {isSelf(m) ? (
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      m.role === 'admin' ? 'bg-purple-900/50 text-purple-300' : 'bg-gray-700 text-gray-300'
                    }`}>
                      {m.role}
                    </span>
                  ) : (
                    <select
                      value={m.role || 'user'}
                      onChange={(e) => handleRoleChange(m, e.target.value)}
                      className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="admin">{t('members.role_admin')}</option>
                      <option value="user">{t('members.role_user')}</option>
                    </select>
                  )}
                </td>
                <td className="px-4 py-3">
                  {m.is_active !== false ? (
                    <span className="flex items-center gap-1 text-green-400 text-xs">
                      <CheckCircle size={14} />
                      {t('members.status_active')}
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400 text-xs">
                      <XCircle size={14} />
                      {t('members.status_inactive')}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs">
                  {m.joined_at ? new Date(m.joined_at).toLocaleDateString() : m.created_at ? new Date(m.created_at).toLocaleDateString() : '-'}
                </td>
                <td className="px-4 py-3 text-right">
                  {!isSelf(m) && (
                    <div className="flex gap-1 justify-end">
                      <button
                        onClick={() => handleToggleActive(m)}
                        className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
                        title={m.is_active !== false ? t('admin.action_deactivate') : t('admin.action_activate')}
                      >
                        {m.is_active !== false ? <ShieldOff size={14} /> : <Shield size={14} />}
                      </button>
                      <button
                        onClick={() => handleRemove(m)}
                        className="p-1.5 rounded hover:bg-red-900/50 text-gray-400 hover:text-red-400"
                        title={t('admin.action_delete_user')}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {members.length === 0 && (
          <div className="text-center text-gray-500 py-8 text-sm">{t('members.no_members')}</div>
        )}
      </div>
    </div>
  );
}
