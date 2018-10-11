using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class Server : MonoBehaviour
{
	private int recv;
	private byte[] _data;
	private IPEndPoint _ipep;
	private Socket _socket;
	private IPEndPoint _sender;
	private EndPoint _remote;
	
	void Start () {
		_ipep = new IPEndPoint(IPAddress.Any, 6969);
		_socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
		
		_socket.Bind(_ipep);
		Debug.Log("Waiting for a client...");
		
		_sender = new IPEndPoint(IPAddress.Any, 0);
		_remote = (EndPoint) (_sender);
	}

	void FixedUpdate () {
		_data = new byte[1024];
		recv = _socket.ReceiveFrom(_data, ref _remote);
	
		Debug.Log(Encoding.ASCII.GetString(_data, 0, recv));
	}
}
